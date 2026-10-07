"""Validate that Ganesha v1 has the data required for a real backtest.

Usage:
    .venv/bin/python scripts/validate_backtest_readiness.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Allow direct execution as ``python scripts/<script>.py`` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        """).execution_options()).one()

        sector_count = db.execute(text("""
            SELECT COUNT(DISTINCT ticker_symbol)
            FROM universe_sector_history
            WHERE effective_from <= :as_of
              AND (effective_to IS NULL OR effective_to > :as_of)
        """), {"as_of": EXPECTED_START}).scalar_one()

        # Delivery data is an end-of-day dataset. The current day's Fyers
        # candle can exist before NSE publishes that day's delivery bhavcopy.
        # For backtest readiness, validate only through the last fully completed
        # day. Any incomplete current-day rows are reported separately and are
        # not used by the backtest.
        last_completed_date = db.execute(text("""
            SELECT MAX(candle_timestamp::date)
            FROM historical_daily_candles
            WHERE delivery_volume IS NOT NULL
              AND delivery_percentage IS NOT NULL
        """)).scalar_one()

        missing_delivery_through_completed = db.execute(text("""
            SELECT COUNT(*)
            FROM historical_daily_candles
            WHERE candle_timestamp::date <= :last_completed_date
              AND (delivery_volume IS NULL OR delivery_percentage IS NULL)
        """), {"last_completed_date": last_completed_date}).scalar_one()

        incomplete_current_day_rows = db.execute(text("""
            SELECT COUNT(*)
            FROM historical_daily_candles
            WHERE candle_timestamp::date > :last_completed_date
              AND (delivery_volume IS NULL OR delivery_percentage IS NULL)
        """), {"last_completed_date": last_completed_date}).scalar_one()

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
        ("Delivery complete through completed day", missing_delivery_through_completed == 0,
         f"missing={missing_delivery_through_completed}, through={last_completed_date}"),
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
    logger.info("Last completed delivery date: {}", last_completed_date)
    if incomplete_current_day_rows:
        logger.warning(
            "Current/incomplete rows excluded from delivery readiness: {}",
            incomplete_current_day_rows,
        )

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
