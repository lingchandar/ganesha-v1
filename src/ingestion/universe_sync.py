"""Daily NSE universe synchronization.

The NSE CM-MII security master is the source of truth for the complete
exchange universe. FYERS remains the live market-data provider and is not
used to define which NSE securities exist.
"""
from datetime import date
from loguru import logger
from sqlalchemy import text

from src.core.database import get_db_session
from src.ingestion.nse_security_master import (
    download_security_master,
    parse_security_master_csv,
    persist_current_security_master,
    persist_security_snapshot,
    save_raw_snapshot,
    security_snapshot_hash,
)


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
        # Store every CM row as evidence. Do not silently reduce this to NIFTY.
        persist_security_snapshot(
            db,
            snapshot_date,
            parsed,
            source="NSE_CM_MII_SECURITY",
            source_file=source_file,
            source_sha256=source_sha256,
            universe_name="NSE_LISTED_CM",
        )

    logger.success(
        "NSE security-master sync complete: {} CM securities stored", len(parsed)
    )
    return len(parsed)


def sync_nse_equity_universe(snapshot_date: date | None = None) -> int:
    """Ingest only EQ-series securities for the stock universe.

    This is a separate resolver from the raw CM master so future eligibility
    rules can exclude ETFs, debt, SME, suspended/illiquid names, etc. without
    losing the raw exchange evidence.
    """
    snapshot_date = snapshot_date or date.today()
    raw, source_file = download_security_master(snapshot_date)
    source_sha256 = security_snapshot_hash(raw)
    parsed = parse_security_master_csv(raw)

    from src.ingestion.nse_security_master import filter_equity_series
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
