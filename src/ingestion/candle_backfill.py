# src/ingestion/candle_backfill.py
"""
GANESHA V1 — Historical Candle Backfill

This script backfills the ``historical_daily_candles`` table with daily OHLCV data
for every active ticker in ``nse_eligible_universe``.

Why Yahoo Finance?
-------------------
* The user's Fyers account is not active yet.
* Yahoo Finance (via ``yfinance``) provides free historical daily candles for the
  NSE with the ``.NS`` suffix. The data quality is sufficient for backtesting and
  ML model training.
* The implementation follows a **DataSourceAdapter** pattern – when the Fyers
  credentials become available you can drop in a ``FyersDataAdapter`` that
  implements the same ``fetch`` interface.

Usage
-----
```bash
# Activate the virtual‑env first if not already active
. .venv/bin/activate
python -m src.ingestion.candle_backfill --years 3
```
The ``--years`` argument controls how many calendar years of data to pull (default
is 3). The script is idempotent – it skips any candle that already exists in the
DB (unique constraint on ``instrument_token`` + ``candle_timestamp``).

Dependencies
------------
* ``yfinance`` – installed in the virtual environment.
* SQLAlchemy session helper ``src.core.database.get_db_session``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import yfinance as yf
from loguru import logger

from src.core.database import get_db_session
from src.ingestion.market_data_client import MarketDataClient  # noqa: F401 (placeholder for future Fyers adapter)
from sqlalchemy import text


class YahooFinanceAdapter:
    """Adapter that fetches daily OHLCV candles for a given NSE ticker.

    yfinance expects the ticker in the form ``<symbol>.NS`` (e.g. ``RELIANCE.NS``).
    The DB stores symbols as ``NSE:<symbol>-EQ``. ``to_yf_ticker`` performs the
    conversion.
    """

    @staticmethod
    def to_yf_ticker(db_ticker: str) -> str:
        # Example: "NSE:BAJAJ-AUTO-EQ" -> "BAJAJ-AUTO.NS"
        if not db_ticker.startswith("NSE:"):
            raise ValueError(f"Unexpected ticker format: {db_ticker}")
        symbol = db_ticker.split(":", 1)[1].replace("-EQ", "")
        return f"{symbol}.NS"

    @staticmethod
    def fetch(symbol: str, start: datetime, end: datetime) -> List[Tuple[datetime, float, float, float, float, int]]:
        """Return a list of daily candles for *symbol* between *start* and *end*.

        Each tuple matches the DB schema order (timestamp, open, high, low, close,
        volume). ``delivery_volume`` and ``delivery_percentage`` are not provided
        by Yahoo Finance and will be stored as ``NULL``.
        """
        yf_ticker = YahooFinanceAdapter.to_yf_ticker(symbol)
        logger.debug(f"Fetching {yf_ticker} from {start.date()} to {end.date()}")
        df = yf.download(yf_ticker, start=start.date(), end=end.date(), interval="1d", progress=False)
        if df.empty:
            logger.warning(f"No data returned for {yf_ticker}")
            return []
        
        # Flatten MultiIndex columns if present in newer yfinance versions
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel('Ticker')

        candles = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime().replace(tzinfo=timezone.utc)
            candles.append(
                (
                    ts,
                    float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"]),
                    float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"]),
                    float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"]),
                    float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"]),
                    int(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else int(row["Volume"]),
                )
            )
        return candles


def backfill_one_symbol(symbol: str, years: int = 3) -> int:
    """Backfill candles for *symbol*.

    Returns the number of rows inserted.
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=years * 365)
    candles = YahooFinanceAdapter.fetch(symbol, start_date, end_date)
    if not candles:
        return 0
    with get_db_session() as db:
        # Resolve the instrument_token for the ticker.
        token_row = db.execute(
            text("SELECT instrument_token FROM nse_eligible_universe WHERE ticker_symbol = :ticker"),
            {"ticker": symbol},
        ).fetchone()
        if token_row is None:
            logger.error(f"Ticker {symbol} not found in nse_eligible_universe – skipping.")
            return 0
        token = token_row[0]
        # Prepare bulk insert with ON CONFLICT DO NOTHING to keep idempotency.
        insert_stmt = text(
            """
            INSERT INTO historical_daily_candles (
                instrument_token,
                candle_timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                volume_traded,
                delivery_volume,
                delivery_percentage,
                is_corporate_action_adjusted
            ) VALUES (
                :instrument_token,
                :candle_timestamp,
                :open_price,
                :high_price,
                :low_price,
                :close_price,
                :volume_traded,
                NULL,
                NULL,
                TRUE
            ) ON CONFLICT (instrument_token, candle_timestamp) DO NOTHING
            """
        )
        inserted = 0
        for ts, o, h, l, c, vol in candles:
            db.execute(
                insert_stmt,
                {
                    "instrument_token": token,
                    "candle_timestamp": ts,
                    "open_price": o,
                    "high_price": h,
                    "low_price": l,
                    "close_price": c,
                    "volume_traded": vol,
                },
            )
            inserted += 1
        db.commit()
        logger.info(f"Inserted {inserted} candles for {symbol} (token {token}).")
        return inserted


def backfill_all(years: int = 3, batch_size: int = 10) -> None:
    """Iterate over active tickers and backfill their historical candles in batches.
    """
    with get_db_session() as db:
        rows = db.execute(
            text("SELECT ticker_symbol, instrument_token FROM nse_eligible_universe WHERE is_active_swing = TRUE ORDER BY instrument_token"),
        ).fetchall()
        universe_map = {r[0]: r[1] for r in rows}
    
    symbols = list(universe_map.keys())
    total_symbols = len(symbols)
    logger.info(f"Starting backfill for {total_symbols} active tickers (last {years} years, batches of {batch_size}).")
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=years * 365)
    
    total_inserted = 0
    success_count = 0
    
    insert_stmt = text(
        """
        INSERT INTO historical_daily_candles (
            instrument_token,
            candle_timestamp,
            open_price,
            high_price,
            low_price,
            close_price,
            volume_traded,
            delivery_volume,
            delivery_percentage,
            is_corporate_action_adjusted
        ) VALUES (
            :instrument_token,
            :candle_timestamp,
            :open_price,
            :high_price,
            :low_price,
            :close_price,
            :volume_traded,
            NULL,
            NULL,
            TRUE
        ) ON CONFLICT (instrument_token, candle_timestamp) DO NOTHING
        """
    )

    for i in range(0, total_symbols, batch_size):
        batch = symbols[i : i + batch_size]
        batch_tickers_yf = [YahooFinanceAdapter.to_yf_ticker(s) for s in batch]
        logger.info(f"Processing batch [{i+1}-{min(i+batch_size, total_symbols)}/{total_symbols}]: {', '.join(batch_tickers_yf)}")
        
        try:
            # Multi-ticker download is faster and respects API limits better
            df = yf.download(batch_tickers_yf, start=start_date.date(), end=end_date.date(), interval="1d", group_by="ticker", progress=False)
            
            with get_db_session() as db:
                for sym in batch:
                    yf_sym = YahooFinanceAdapter.to_yf_ticker(sym)
                    token = universe_map[sym]
                    
                    if len(batch_tickers_yf) == 1:
                        sym_df = df
                    else:
                        if yf_sym not in df.columns.levels[0]:
                            continue
                        sym_df = df[yf_sym]
                    
                    if sym_df.empty:
                        continue
                        
                    sym_inserted = 0
                    for idx, row in sym_df.iterrows():
                        if row["Close"] is None or str(row["Close"]).lower() == "nan":
                            continue
                        ts = idx.to_pydatetime().replace(tzinfo=timezone.utc)
                        o = float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"])
                        h = float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"])
                        l = float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"])
                        c = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
                        vol = int(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else int(row["Volume"])
                        
                        db.execute(
                            insert_stmt,
                            {
                                "instrument_token": token,
                                "candle_timestamp": ts,
                                "open_price": o,
                                "high_price": h,
                                "low_price": l,
                                "close_price": c,
                                "volume_traded": vol,
                            },
                        )
                        sym_inserted += 1
                    
                    total_inserted += sym_inserted
                    success_count += 1
                db.commit()
                
        except Exception as exc:
            logger.warning(f"Batch download error ({exc}), falling back to single-ticker fetch...")
            for sym in batch:
                try:
                    count = backfill_one_symbol(sym, years=years)
                    total_inserted += count
                    if count > 0:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed fallback backfill for {sym}: {e}")

    logger.success(f"Backfill finished: {success_count}/{total_symbols} stocks processed, {total_inserted} total candles stored.")


def seed_derivative_expiry_calendar(start_year: int = datetime.now().year, end_year: int = datetime.now().year + 2) -> None:
    """Populate ``nse_derivative_expiry_calendar`` with monthly expiry dates.

    NSE monthly futures expire on the last Thursday of the month. For simplicity we
    generate the *contract month* as ``YYYYMM`` and calculate a three‑day buffer
    start (``expiry_date - 3 days``) adjusted for weekends.
    """
    from calendar import monthrange
    from dateutil.relativedelta import relativedelta

    dates: List[Tuple[datetime, str, datetime]] = []
    current = datetime(start_year, 1, 1)
    while current.year <= end_year:
        # Find the last Thursday of the month.
        last_day = monthrange(current.year, current.month)[1]
        last_date = datetime(current.year, current.month, last_day)
        # Backtrack to Thursday.
        while last_date.weekday() != 3:  # 0=Mon … 3=Thu
            last_date -= timedelta(days=1)
        expiry = last_date.date()
        buffer_start = expiry - timedelta(days=3)
        # If buffer_start lands on weekend, push to previous Friday.
        if buffer_start.weekday() >= 5:
            buffer_start -= timedelta(days=buffer_start.weekday() - 4)
        contract_month = f"{current.year}{current.month:02d}"
        dates.append((expiry, contract_month, buffer_start))
        # Move to next month.
        current += relativedelta(months=1)

    with get_db_session() as db:
        insert_stmt = text(
            """
            INSERT INTO nse_derivative_expiry_calendar (
                expiry_date,
                contract_month,
                is_monthly_expiry,
                three_day_buffer_start,
                created_at
            ) VALUES (
                :expiry_date,
                :contract_month,
                TRUE,
                :buffer_start,
                CURRENT_TIMESTAMP
            ) ON CONFLICT (expiry_date) DO UPDATE SET
                contract_month = EXCLUDED.contract_month,
                three_day_buffer_start = EXCLUDED.three_day_buffer_start
            """
        )
        for expiry, cm, buf in dates:
            db.execute(
                insert_stmt,
                {"expiry_date": expiry, "contract_month": cm, "buffer_start": buf},
            )
        db.commit()
    logger.success(f"Seeded {len(dates)} monthly expiry rows into nse_derivative_expiry_calendar.")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill historical candles & seed expiry calendar.")
    parser.add_argument("--years", type=int, default=3, help="Number of calendar years of data to fetch.")
    parser.add_argument("--seed-expiry", action="store_true", help="Populate the derivative expiry calendar (runs after backfill).")
    args = parser.parse_args(argv)

    backfill_all(years=args.years)
    if args.seed_expiry:
        seed_derivative_expiry_calendar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
