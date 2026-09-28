"""
GANESHA V1 — Historical Candle Backfill (Fyers API v3)
Usage:
    .venv/bin/python -m src.ingestion.candle_backfill --years 3
    .venv/bin/python -m src.ingestion.candle_backfill --years 3 --seed-expiry
Requires today's token: .venv/bin/python -m src.ingestion.fyers_auth
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from typing import List, Tuple

from loguru import logger
from sqlalchemy import text

from src.core.database import get_db_session
from src.ingestion.market_data_client import MarketDataClient

CHUNK_DAYS = 360  # Fyers caps daily-resolution requests at ~366 days

INSERT_SQL = text("""
    INSERT INTO historical_daily_candles (
        instrument_token, candle_timestamp, open_price, high_price, low_price,
        close_price, volume_traded, delivery_volume, delivery_percentage,
        is_corporate_action_adjusted
    ) VALUES (
        :token, :ts, :o, :h, :l, :c, :v, NULL, NULL, TRUE
    ) ON CONFLICT (instrument_token, candle_timestamp) DO UPDATE SET
        open_price = EXCLUDED.open_price,
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        volume_traded = EXCLUDED.volume_traded
""")


def fetch_symbol(client: MarketDataClient, symbol: str, start: date, end: date) -> List[dict]:
    """Fetch daily candles for one symbol, in <=360-day chunks."""
    rows, cur = [], start
    while cur < end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        df = client.fetch_historical_candles(symbol, cur, chunk_end)
        for _, r in df.iterrows():
            d = r["timestamp"].date()
            rows.append({
                "ts": datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
                "o": float(r["open_price"]),
                "h": float(r["high_price"]),
                "l": float(r["low_price"]),
                "c": float(r["close_price"]),
                "v": int(r["volume_traded"]),
            })
        cur = chunk_end + timedelta(days=1)
    return rows


def backfill_all(years: int = 3) -> None:
    client = MarketDataClient()
    if not client.authenticate():
        logger.error("No valid Fyers token today. Run: .venv/bin/python -m src.ingestion.fyers_auth")
        sys.exit(1)

    with get_db_session() as db:
        universe = db.execute(text(
            "SELECT ticker_symbol, instrument_token FROM nse_eligible_universe "
            "WHERE is_active_swing = TRUE ORDER BY instrument_token"
        )).fetchall()

    end = date.today()
    start = end - timedelta(days=years * 365)
    total, ok, failed = 0, 0, []
    logger.info(f"Backfilling {len(universe)} tickers from Fyers ({start} to {end})")

    for i, (symbol, token) in enumerate(universe, 1):
        try:
            rows = fetch_symbol(client, symbol, start, end)
            if not rows:
                failed.append(symbol)
                logger.warning(f"[{i}/{len(universe)}] {symbol}: no candles returned")
                continue
            for r in rows:
                r["token"] = token
            with get_db_session() as db:
                db.execute(INSERT_SQL, rows)
            total += len(rows)
            ok += 1
            logger.info(f"[{i}/{len(universe)}] {symbol}: {len(rows)} candles")
        except Exception as e:
            failed.append(symbol)
            logger.error(f"[{i}/{len(universe)}] {symbol} failed: {e}")

    client.close()
    logger.success(f"Done: {ok}/{len(universe)} tickers, {total} candles stored.")
    if failed:
        logger.warning(f"No data for {len(failed)} tickers: {', '.join(failed)}")


def seed_derivative_expiry_calendar(start_year: int = datetime.now().year,
                                    end_year: int = datetime.now().year + 2) -> None:
    """Populate nse_derivative_expiry_calendar with last-Thursday monthly expiries."""
    from calendar import monthrange

    rows: List[Tuple[date, str, date]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, monthrange(year, month)[1])
            while d.weekday() != 3:
                d -= timedelta(days=1)
            buf = d - timedelta(days=3)
            if buf.weekday() >= 5:
                buf -= timedelta(days=buf.weekday() - 4)
            rows.append((d, f"{year}{month:02d}", buf))

    with get_db_session() as db:
        for expiry, cm, buf in rows:
            db.execute(text("""
                INSERT INTO nse_derivative_expiry_calendar
                    (expiry_date, contract_month, is_monthly_expiry, three_day_buffer_start, created_at)
                VALUES (:e, :cm, TRUE, :b, CURRENT_TIMESTAMP)
                ON CONFLICT (expiry_date) DO UPDATE SET
                    contract_month = EXCLUDED.contract_month,
                    three_day_buffer_start = EXCLUDED.three_day_buffer_start
            """), {"e": expiry, "cm": cm, "b": buf})
    logger.success(f"Seeded {len(rows)} monthly expiry rows.")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill candles from Fyers.")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--seed-expiry", action="store_true")
    args = p.parse_args(argv)
    backfill_all(years=args.years)
    if args.seed_expiry:
        seed_derivative_expiry_calendar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    