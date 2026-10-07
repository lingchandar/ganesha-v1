"""Backfill official NSE delivery data into historical daily candles.

The Fyers candle feed supplies OHLCV but not NSE delivery quantities. The
backtest delivery gate therefore needs a separate historical source. This
module downloads each available NSE security-wise delivery bhavcopy once per
trading date and updates only the controlled universe's existing candle rows.

NSE symbols can change over time. The controlled universe intentionally uses
current symbols, so historical delivery rows are resolved through explicit
NSE rename aliases when the current symbol is absent from an older bhavcopy.

Usage:
    .venv/bin/python -m src.ingestion.delivery_backfill
    .venv/bin/python -m src.ingestion.delivery_backfill --start-date 2023-10-09
    .venv/bin/python -m src.ingestion.delivery_backfill --limit-days 5
"""
from __future__ import annotations

import argparse
from datetime import date
from typing import List

from loguru import logger
from sqlalchemy import text

from src.core.database import get_db_session
from src.ingestion.nse_delivery_client import NSEArchiveDeliveryClient

UPDATE_SQL = text("""
    UPDATE historical_daily_candles
    SET delivery_volume = :delivery_volume,
        delivery_percentage = :delivery_percentage
    WHERE ticker_symbol = :ticker_symbol
      AND candle_timestamp::date = :trade_date
""")

# Current controlled-universe symbols -> historical NSE symbols.
# These are required because the candle universe keeps the current symbol while
# NSE delivery bhavcopies use the symbol that was valid on the trade date.
HISTORICAL_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "ETERNAL": ("ZOMATO",),
    "TMPV": ("TATAMOTORS",),
}


def _load_trade_dates(start_date: date | None, end_date: date | None) -> List[date]:
    """Read trading dates already present in the candle table."""
    clauses = []
    params: dict[str, date] = {}
    if start_date:
        clauses.append("candle_timestamp::date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        clauses.append("candle_timestamp::date <= :end_date")
        params["end_date"] = end_date

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = text(f"""
        SELECT DISTINCT candle_timestamp::date AS trade_date
        FROM historical_daily_candles
        {where}
        ORDER BY trade_date
    """)
    with get_db_session() as db:
        return [row.trade_date for row in db.execute(query, params).fetchall()]


def _load_controlled_symbols() -> set[str]:
    with get_db_session() as db:
        rows = db.execute(text("""
            SELECT DISTINCT ticker_symbol
            FROM universe_membership_history
            WHERE universe_name = 'NSE_SWING'
              AND effective_to IS NULL
        """)).fetchall()
    return {row.ticker_symbol.removeprefix("NSE:").removesuffix("-EQ") for row in rows}


def _resolve_delivery_rows(df, controlled_symbols: set[str], trade_date: date):
    """Map historical NSE symbols to the current controlled-universe symbols.

    Prefer an exact current symbol. Only when it is absent do we use an
    explicitly configured historical alias. This prevents an old alias from
    overriding a valid current-symbol record after a rename.
    """
    rows = []
    available = set(df["ticker_symbol"].astype(str))

    for symbol in sorted(controlled_symbols):
        source_symbol = symbol
        if source_symbol not in available:
            for alias in HISTORICAL_SYMBOL_ALIASES.get(symbol, ()):
                if alias in available:
                    source_symbol = alias
                    break
            else:
                continue

        match = df[df["ticker_symbol"] == source_symbol]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "ticker_symbol": f"NSE:{symbol}-EQ",
                "trade_date": trade_date,
                "delivery_volume": int(row.delivery_volume),
                "delivery_percentage": float(row.delivery_percentage),
            }
        )

    return rows


def backfill_delivery(
    start_date: date | None = None,
    end_date: date | None = None,
    limit_days: int | None = None,
) -> None:
    trade_dates = _load_trade_dates(start_date, end_date)
    if limit_days is not None:
        trade_dates = trade_dates[:limit_days]

    if not trade_dates:
        raise RuntimeError("No historical candle dates found for delivery backfill")

    controlled_symbols = _load_controlled_symbols()
    if not controlled_symbols:
        raise RuntimeError("NSE_SWING controlled universe is empty")

    client = NSEArchiveDeliveryClient()
    processed = 0
    updated_rows = 0
    missing_dates = 0
    failed_dates: list[str] = []

    try:
        logger.info(
            "Backfilling NSE delivery for {} trading dates ({} to {}) and {} controlled symbols",
            len(trade_dates), trade_dates[0], trade_dates[-1], len(controlled_symbols),
        )

        for index, trade_date in enumerate(trade_dates, start=1):
            try:
                df = client.fetch_daily_delivery_bhavcopy(trade_date)
                if df.empty:
                    missing_dates += 1
                    logger.warning("[{}/{}] {}: no NSE delivery file", index, len(trade_dates), trade_date)
                    continue

                rows = _resolve_delivery_rows(df, controlled_symbols, trade_date)
                if not rows:
                    missing_dates += 1
                    logger.warning("[{}/{}] {}: no controlled symbols in NSE file", index, len(trade_dates), trade_date)
                    continue

                with get_db_session() as db:
                    result = db.execute(UPDATE_SQL, rows)
                    updated = int(result.rowcount or 0)

                processed += 1
                updated_rows += updated
                logger.info(
                    "[{}/{}] {}: {} resolved NSE rows, {} candle rows updated",
                    index, len(trade_dates), trade_date, len(rows), updated,
                )
            except Exception as exc:
                failed_dates.append(trade_date.isoformat())
                logger.error("[{}/{}] {} failed: {}", index, len(trade_dates), trade_date, exc)
    finally:
        client.close()

    logger.success(
        "Delivery backfill complete: {}/{} dates processed, {} candle rows updated, {} missing dates, {} failed dates",
        processed, len(trade_dates), updated_rows, missing_dates, len(failed_dates),
    )
    if failed_dates:
        logger.warning("Failed dates: {}", ", ".join(failed_dates[:20]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill official NSE delivery data")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--limit-days", type=int)
    args = parser.parse_args()
    backfill_delivery(args.start_date, args.end_date, args.limit_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
