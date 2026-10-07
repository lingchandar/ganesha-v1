"""Validate the quality and coverage of historical daily candles."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.database import get_db_session  # noqa: E402


def main() -> int:
    with get_db_session() as session:
        universe = session.execute(
            text(
                """
                SELECT DISTINCT instrument_token, symbol
                FROM universe_membership_history
                WHERE universe_name = 'NSE_SWING'
                  AND valid_to IS NULL
                ORDER BY symbol
                """
            )
        ).mappings().all()

        if not universe:
            print("ERROR: NSE_SWING universe is empty")
            return 1

        total = session.execute(
            text("SELECT COUNT(*) FROM historical_daily_candles")
        ).scalar_one()

        invalid = session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM historical_daily_candles
                WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                   OR high < low
                   OR high < open OR high < close
                   OR low > open OR low > close
                   OR volume < 0
                """
            )
        ).scalar_one()

        duplicate_groups = session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT instrument_token, candle_timestamp
                    FROM historical_daily_candles
                    GROUP BY instrument_token, candle_timestamp
                    HAVING COUNT(*) > 1
                ) duplicates
                """
            )
        ).scalar_one()

        print(f"Universe: {len(universe)} stocks")
        print(f"Total candles: {total}")
        print(f"Invalid OHLC/volume rows: {invalid}")
        print(f"Duplicate timestamp groups: {duplicate_groups}")
        print()
        print("Per-stock coverage:")
        print("SYMBOL\tROWS\tFIRST_DATE\tLAST_DATE\tGAPS_GT_7D")

        failures = 0
        for item in universe:
            rows = session.execute(
                text(
                    """
                    SELECT COUNT(*) AS row_count,
                           MIN(candle_timestamp)::date AS first_date,
                           MAX(candle_timestamp)::date AS last_date
                    FROM historical_daily_candles
                    WHERE instrument_token = :instrument_token
                    """
                ),
                {"instrument_token": item["instrument_token"]},
            ).mappings().one()

            gaps = session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT candle_timestamp::date AS d,
                               LAG(candle_timestamp::date) OVER (
                                   ORDER BY candle_timestamp
                               ) AS prev_d
                        FROM historical_daily_candles
                        WHERE instrument_token = :instrument_token
                    ) x
                    WHERE prev_d IS NOT NULL
                      AND d - prev_d > 7
                    """
                ),
                {"instrument_token": item["instrument_token"]},
            ).scalar_one()

            print(
                f"{item['symbol']}\t{rows['row_count']}\t"
                f"{rows['first_date']}\t{rows['last_date']}\t{gaps}"
            )
            if rows["row_count"] == 0 or gaps > 0:
                failures += 1

        print()
        if invalid or duplicate_groups or failures:
            print("VALIDATION: FAIL")
            return 1

        print("VALIDATION: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
