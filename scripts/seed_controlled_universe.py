"""Seed Ganesha v1's controlled NSE swing universe.

The v1 live scanner intentionally uses the union of current NIFTY 50 and
BANKNIFTY constituents. This script keeps that scope separate from the legacy
NIFTY-100 seeder and creates the point-in-time membership and sector records
required by the backtest loader.

For v1 backtesting, this is deliberately modeled as a fixed controlled
universe beginning at the start of the available three-year candle dataset.
This is NOT a reconstruction of historical NIFTY 50/BANKNIFTY membership.
It avoids survivorship ambiguity while we validate the strategy pipeline.

Instrument tokens are internal stable IDs for Ganesha's candle key. FYERS
requests use ticker_symbol (NSE:SYMBOL-EQ), never this internal token.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys

# Allow direct execution from the repository's scripts/ directory:
# `.venv/bin/python scripts/seed_controlled_universe.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from sqlalchemy import bindparam, text

from src.core.database import get_db_session
from src.ingestion.nse_index_universe import load_controlled_index_universe

# The candle backfill currently covers this three-year window. Keep the
# controlled v1 universe active from its first historical trading date so the
# backtest loader can resolve membership/sector data before today's seed date.
CONTROLLED_UNIVERSE_EFFECTIVE_FROM = date(2023, 10, 9)

# Broad sector labels used by the risk/backtest layers. Unknown symbols fail
# closed into OTHER rather than preventing the 59-stock universe from loading.
SECTOR_BY_SYMBOL = {
    # Financial services / banks
    "AUBANK": "FINANCIAL_SERVICES",
    "AXISBANK": "FINANCIAL_SERVICES",
    "BANKBARODA": "FINANCIAL_SERVICES",
    "BANDHANBNK": "FINANCIAL_SERVICES",
    "CANBK": "FINANCIAL_SERVICES",
    "FEDERALBNK": "FINANCIAL_SERVICES",
    "HDFCBANK": "FINANCIAL_SERVICES",
    "ICICIBANK": "FINANCIAL_SERVICES",
    "IDFCFIRSTB": "FINANCIAL_SERVICES",
    "INDUSINDBK": "FINANCIAL_SERVICES",
    "KOTAKBANK": "FINANCIAL_SERVICES",
    "PNB": "FINANCIAL_SERVICES",
    "SBIN": "FINANCIAL_SERVICES",
    "SHRIRAMFIN": "FINANCIAL_SERVICES",
    "BAJFINANCE": "FINANCIAL_SERVICES",
    # IT
    "HCLTECH": "INFORMATION_TECHNOLOGY",
    "INFY": "INFORMATION_TECHNOLOGY",
    "TCS": "INFORMATION_TECHNOLOGY",
    "TECHM": "INFORMATION_TECHNOLOGY",
    "WIPRO": "INFORMATION_TECHNOLOGY",
    # Energy / oil & gas
    "BPCL": "OIL_AND_GAS",
    "COALINDIA": "OIL_AND_GAS",
    "ONGC": "OIL_AND_GAS",
    "RELIANCE": "OIL_AND_GAS",
    # Automobiles
    "EICHERMOT": "AUTOMOBILE",
    "HEROMOTOCO": "AUTOMOBILE",
    "M&M": "AUTOMOBILE",
    "MARUTI": "AUTOMOBILE",
    "TATAMOTORS": "AUTOMOBILE",
    # FMCG / consumer
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "TATACONSUM": "FMCG",
    "TITAN": "CONSUMER_SERVICES",
    # Healthcare
    "APOLLOHOSP": "HEALTHCARE",
    "CIPLA": "HEALTHCARE",
    "DRREDDY": "HEALTHCARE",
    "SUNPHARMA": "HEALTHCARE",
    # Metals
    "HINDALCO": "METALS",
    "JSWSTEEL": "METALS",
    "TATASTEEL": "METALS",
    # Capital goods / infrastructure
    "ADANIPORTS": "CAPITAL_GOODS",
    "BEL": "CAPITAL_GOODS",
    "HAL": "CAPITAL_GOODS",
    "LT": "CAPITAL_GOODS",
    # Power / utilities
    "NTPC": "POWER",
    "POWERGRID": "POWER",
    # Telecom
    "BHARTIARTL": "TELECOM",
    # Cement
    "GRASIM": "CEMENT",
    "ULTRACEMCO": "CEMENT",
}


def seed_controlled_universe() -> int:
    symbols = load_controlled_index_universe()
    today = date.today()
    now = datetime.now(timezone.utc)

    if len(symbols) != 59:
        raise RuntimeError(
            f"Expected 59 unique NIFTY50 + BANKNIFTY symbols, got {len(symbols)}"
        )

    with get_db_session() as db:
        # Repair the original seed, which incorrectly started membership at
        # today. Move that open-ended row to the historical start date rather
        # than creating overlapping open-ended membership periods.
        db.execute(
            text("""
                UPDATE universe_membership_history
                SET effective_from = :start_date
                WHERE universe_name = 'NSE_SWING'
                  AND effective_from = :today
                  AND effective_to IS NULL
            """),
            {"start_date": CONTROLLED_UNIVERSE_EFFECTIVE_FROM, "today": today},
        )
        db.execute(
            text("""
                UPDATE universe_sector_history
                SET effective_from = :start_date
                WHERE effective_from = :today
            """),
            {"start_date": CONTROLLED_UNIVERSE_EFFECTIVE_FROM, "today": today},
        )

        # Close memberships that are no longer part of today's controlled set.
        db.execute(
            text("""
                UPDATE universe_membership_history
                SET effective_to = :today
                WHERE universe_name = 'NSE_SWING'
                  AND effective_to IS NULL
                  AND effective_from < :today
                  AND ticker_symbol NOT IN :symbols
            """).bindparams(bindparam("symbols", expanding=True)),
            {"today": today, "symbols": list(symbols)},
        )

        for token, fyers_symbol in enumerate(symbols, start=20001):
            raw_symbol = fyers_symbol.removeprefix("NSE:").removesuffix("-EQ")
            sector = SECTOR_BY_SYMBOL.get(raw_symbol, "OTHER")
            industry = "BANKING" if sector == "FINANCIAL_SERVICES" else None

            db.execute(
                text("""
                    INSERT INTO universe_membership_history
                        (universe_name, ticker_symbol, instrument_token, effective_from)
                    VALUES ('NSE_SWING', :ticker, :token, :effective_from)
                    ON CONFLICT (universe_name, ticker_symbol, effective_from)
                    DO UPDATE SET instrument_token = EXCLUDED.instrument_token,
                                  effective_to = NULL
                """),
                {
                    "ticker": fyers_symbol,
                    "token": token,
                    "effective_from": CONTROLLED_UNIVERSE_EFFECTIVE_FROM,
                },
            )

            db.execute(
                text("""
                    INSERT INTO nse_eligible_universe
                        (instrument_token, ticker_symbol, company_name, sector_name,
                         industry_name, is_active_swing, last_updated_at)
                    VALUES (:token, :ticker, :company, :sector, :industry, TRUE, :now)
                    ON CONFLICT (ticker_symbol) DO UPDATE SET
                        instrument_token = EXCLUDED.instrument_token,
                        company_name = EXCLUDED.company_name,
                        sector_name = EXCLUDED.sector_name,
                        industry_name = EXCLUDED.industry_name,
                        is_active_swing = TRUE,
                        last_updated_at = EXCLUDED.last_updated_at
                """),
                {
                    "token": token,
                    "ticker": fyers_symbol,
                    "company": raw_symbol,
                    "sector": sector,
                    "industry": industry,
                    "now": now,
                },
            )

            db.execute(
                text("""
                    INSERT INTO universe_sector_history
                        (ticker_symbol, sector_name, industry_name, effective_from)
                    VALUES (:ticker, :sector, :industry, :effective_from)
                    ON CONFLICT (ticker_symbol, effective_from) DO UPDATE SET
                        sector_name = EXCLUDED.sector_name,
                        industry_name = EXCLUDED.industry_name,
                        source_updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "ticker": fyers_symbol,
                    "sector": sector,
                    "industry": industry,
                    "effective_from": CONTROLLED_UNIVERSE_EFFECTIVE_FROM,
                },
            )

    logger.success(
        "Seeded controlled Ganesha v1 universe: {} symbols from {}",
        len(symbols),
        CONTROLLED_UNIVERSE_EFFECTIVE_FROM,
    )
    return len(symbols)


if __name__ == "__main__":
    seed_controlled_universe()
