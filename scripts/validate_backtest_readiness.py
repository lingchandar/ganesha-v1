"""Validate that Ganesha v1 has the data required for a real backtest.

Usage:
    .venv/bin/python scripts/validate_backtest_readiness.py
"""
from __future__ import annotations

from datetime import date

from loguru import logger
from sqlalchemy import text

from src.core.database import get_db_session

EXPECTED_UNIVERSE = 59
EXPECTED_START = date(2023, 10, 9)


def main() -> int:
    with get_db_session() as db:
        universe_count = db.execute(text("""
            SELECT COUNT(DISTINCT ticker_symbol)
            FROM universe_membership_history
            WHERE universe_name = 'NSE_SWING'
        """)).scalar_one()

        membership_start = db.execute(text("""
            SELECT MIN(effective_from)
            FROM universe_membership_history
            WHERE universe_name = 'NSE_SWING'
        """)).scalar_one()

        candle_stats = db.execute(text("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT ticker_symbol) AS symbols,
                   MIN(candle_timestamp::date) AS first_date,
                   MAX(candle_timestamp::date) AS last_date
            FROM historical_daily_candles
        """)).one()

        delivery_stats = db.execute(text("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT ticker_symbol) AS symbols,
                   COUNT(*) FILTER (WHERE delivery_volume IS NOT NULL
                                    AND delivery_percentage IS NOT NULL) AS complete_rows
            FROM historical_daily_candles
        """)).one()

        sector_count = db.execute(text("""
            SELECT COUNT(DISTINCT ticker_symbol)
            FROM universe_sector_history
            WHERE effective_from <= :as_of
              AND (effective_to IS NULL OR effective_to > :as_of)
        """), {"as_of": EXPECTED_START}).scalar_one()

        missing_delivery = db.execute(text("""
            SELECT COUNT(*)
            FROM historical_daily_candles
            WHERE delivery_volume IS NULL OR delivery_percentage IS NULL
        """)).scalar_one()

    checks = [
        ("Universe symbols", universe_count == EXPECTED_UNIVERSE,
         f"{universe_count}/{EXPECTED_UNIVERSE}"),
        ("Historical membership start", membership_start == EXPECTED_START,
         str(membership_start)),
        ("Candle symbols", candle_stats.symbols == EXPECTED_UNIVERSE,
         f"{candle_stats.symbols}/{EXPECTED_UNIVERSE}"),
        ("Candle rows", candle_stats.rows > 0, str(candle_stats.rows)),
        ("Candle start", candle_stats.first_date <= EXPECTED_START,
         str(candle_stats.first_date)),
        ("Delivery complete", missing_delivery == 0,
         f"missing={missing_delivery}"),
        ("Sector history", sector_count == EXPECTED_UNIVERSE,
         f"{sector_count}/{EXPECTED_UNIVERSE}"),
    ]

    failed = False
    logger.info("Backtest readiness")
    logger.info("Universe: {} symbols", universe_count)
    logger.info("Candles: {} rows, {} symbols, {} to {}",
                candle_stats.rows, candle_stats.symbols,
                candle_stats.first_date, candle_stats.last_date)
    logger.info("Delivery: {} complete rows / {} candle rows",
                delivery_stats.complete_rows, delivery_stats.rows)

    for name, passed, value in checks:
        if passed:
            logger.success("PASS  {}: {}", name, value)
        else:
            failed = True
            logger.error("FAIL  {}: {}", name, value)

    if failed:
        logger.error("BACKTEST READINESS: FAIL")
        return 1

    logger.success("BACKTEST READINESS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
