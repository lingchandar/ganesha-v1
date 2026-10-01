"""Daily NSE universe synchronization.

The NSE CM-MII security master is the source of truth for the complete
exchange universe. FYERS remains the live market-data provider and is not
used to define which NSE securities exist.
"""
from datetime import date
import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.core.database import get_db_session
from src.ingestion.liquidity_filter import LiquidityConfig, evaluate_liquidity
from src.ingestion.nse_security_master import (
    download_security_master,
    parse_security_master_csv,
    persist_current_security_master,
    persist_security_snapshot,
    save_raw_snapshot,
    security_snapshot_hash,
    filter_equity_series,
    to_fyers_equity_symbol,
)


def build_swing_universe(
    membership,
    candles_by_ticker,
    config: LiquidityConfig = LiquidityConfig(),
):
    """Apply the point-in-time liquidity gate to NSE stock membership.

    Missing candle history fails closed. No future observations are used:
    callers must provide candles only through the evaluation date.
    """
    rows = []
    for row in membership.to_dict("records"):
        ticker = row["ticker_symbol"]
        candles = candles_by_ticker.get(ticker)
        result = (
            evaluate_liquidity(candles, config)
            if candles is not None
            else {
                "eligible": False,
                "reason": "MISSING_LIQUIDITY_HISTORY",
                "observations": 0,
            }
        )
        rows.append({**row, **result})
    return pd.DataFrame(rows)


def sync_nse_security_master(snapshot_date: date | None = None) -> int:
    """Ingest one official NSE CM security-master snapshot.

    No NIFTY index membership is used here. The complete NSE CM file is
    retained first; swing/liquidity eligibility is a later stage.
    """
    snapshot_date = snapshot_date or date.today()
    logger.info("Starting complete NSE security-master sync for {}", snapshot_date)

    raw, source_file = download_security_master(snapshot_date)
    source_sha256 = security_snapshot_hash(raw)
    parsed = parse_security_master_csv(raw)

    with get_db_session() as db:
        save_raw_snapshot(raw, snapshot_date, "data/nse_security_master")
        persist_current_security_master(
            db,
            snapshot_date,
            parsed,
            source="NSE_CM_MII_SECURITY",
            source_file=source_file,
            source_sha256=source_sha256,
        )
        persist_security_snapshot(
            db,
            snapshot_date,
            parsed,
            source="NSE_CM_MII_SECURITY",
            source_file=source_file,
            source_sha256=source_sha256,
            universe_name="NSE_LISTED_CM",
        )

        equity = filter_equity_series(parsed)
        current_tickers = set()
        for row in equity.to_dict("records"):
            # Keep the NSE security ID as internal evidence, but store the
            # exchange-qualified symbol used by FYERS for market-data calls.
            fyers_symbol = to_fyers_equity_symbol(row["symbol"])
            current_tickers.add(fyers_symbol)
            db.execute(
                text("""
                    INSERT INTO universe_membership_history
                        (universe_name, ticker_symbol, instrument_token, effective_from)
                    VALUES
                        ('NSE_SWING', :ticker, :token, :effective_from)
                    ON CONFLICT (universe_name, ticker_symbol, effective_from)
                    DO UPDATE SET instrument_token = EXCLUDED.instrument_token
                """),
                {
                    "ticker": fyers_symbol,
                    "token": int(row["instrument_token"])
                    if row.get("instrument_token") is not None else 0,
                    "effective_from": snapshot_date,
                },
            )

        db.execute(
            text("""
                UPDATE universe_membership_history
                SET effective_to = :effective_to
                WHERE universe_name = 'NSE_SWING'
                  AND effective_to IS NULL
                  AND effective_from < :snapshot_date
                  AND ticker_symbol NOT IN :tickers
            """).bindparams(__import__("sqlalchemy").bindparam("tickers", expanding=True)),
            {"effective_to": snapshot_date, "snapshot_date": snapshot_date, "tickers": list(current_tickers)},
        )

    logger.success(
        "NSE security-master sync complete: {} CM securities stored", len(parsed)
    )
    return len(parsed)


def sync_nse_equity_universe(snapshot_date: date | None = None) -> int:
    """Ingest the NSE stock universe for the swing scanner.

    The resolver includes NSE equity/SME series while excluding Rights
    Entitlements, ETFs and test securities. Liquidity and swing rules remain
    separate so the raw exchange evidence is never lost.
    """
    snapshot_date = snapshot_date or date.today()
    raw, source_file = download_security_master(snapshot_date)
    source_sha256 = security_snapshot_hash(raw)
    parsed = parse_security_master_csv(raw)

    equity = filter_equity_series(parsed)

    with get_db_session() as db:
        persist_security_snapshot(
            db,
            snapshot_date,
            equity,
            source="NSE_CM_MII_SECURITY",
            source_file=source_file,
            source_sha256=source_sha256,
            universe_name="NSE_LISTED_EQUITY",
        )

    logger.success(
        "NSE EQ universe snapshot complete: {} equity-series securities", len(equity)
    )
    return len(equity)


if __name__ == "__main__":
    sync_nse_security_master()

def load_current_swing_symbols(db, as_of: date | None = None) -> list[str]:
    """Load point-in-time FYERS symbols currently in the NSE_SWING membership."""
    as_of = as_of or date.today()
    rows = db.execute(
        text("""
            SELECT ticker_symbol
            FROM universe_membership_history
            WHERE universe_name = 'NSE_SWING'
              AND effective_from <= :as_of
              AND (effective_to IS NULL OR effective_to > :as_of)
            ORDER BY ticker_symbol
        """),
        {"as_of": as_of},
    ).scalars().all()
    return [str(symbol) for symbol in rows if str(symbol).strip()]

