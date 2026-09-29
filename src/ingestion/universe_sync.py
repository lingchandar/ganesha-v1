"""
GANESHA V1 — Universe Sync: Fetches and stores the active NIFTY 100 constituent list.

Runs on the 1st trading session of every month to refresh the eligible universe.
Upserts into nse_eligible_universe table, marking removed stocks as inactive.
"""
from datetime import datetime
from sqlalchemy import text
from loguru import logger

from src.core.database import get_db_session
from src.ingestion.market_data_client import MarketDataClient


def sync_nifty100_universe() -> int:
    """
    Fetch the latest NIFTY 50 + NIFTY NEXT 50 constituents
    and upsert them into the nse_eligible_universe table.

    Returns:
        Number of stocks synced.
    """
    logger.info("Starting NIFTY 100 universe sync...")

    client = MarketDataClient()
    synced_count = 0

    try:
        client.authenticate()
        symbols = client.fetch_nifty100_symbols()
        logger.info(f"Fetched {len(symbols)} symbols from broker API")

        with get_db_session() as db:
            # Step 1: Mark ALL existing stocks as inactive
            db.execute(
                text("UPDATE nse_eligible_universe SET is_active_swing = FALSE")
            )

            # Step 2: Upsert each fetched symbol (re-activating current constituents)
            for sym in symbols:
                db.execute(
                    text("""
                        INSERT INTO nse_eligible_universe
                            (instrument_token, ticker_symbol, company_name,
                             sector_name, industry_name, is_active_swing, last_updated_at)
                        VALUES
                            (:token, :ticker, :name, :sector, :industry, TRUE, :now)
                        ON CONFLICT (ticker_symbol)
                        DO UPDATE SET
                            instrument_token = EXCLUDED.instrument_token,
                            sector_name = EXCLUDED.sector_name,
                            industry_name = EXCLUDED.industry_name,
                            is_active_swing = TRUE,
                            last_updated_at = EXCLUDED.last_updated_at
                    """),
                    {
                        "token": sym["token"],
                        "ticker": sym["symbol"],
                        "name": sym["name"],
                        "sector": sym["sector"],
                        "industry": sym.get("industry", ""),
                        "now": datetime.now(),
                    },
                )
                # Preserve classification history instead of relying on the mutable master row.
                sync_date = datetime.now().date()
                previous = db.execute(text("""
                    SELECT sector_name, industry_name
                    FROM universe_sector_history
                    WHERE ticker_symbol = :ticker AND effective_to IS NULL
                    ORDER BY effective_from DESC LIMIT 1
                """), {"ticker": sym["symbol"]}).mappings().first()
                if previous is None:
                    db.execute(text("""
                        INSERT INTO universe_sector_history
                            (ticker_symbol, sector_name, industry_name, effective_from)
                        VALUES (:ticker, :sector, :industry, :effective_from)
                    """), {"ticker": sym["symbol"], "sector": sym["sector"],
                           "industry": sym.get("industry", ""), "effective_from": sync_date})
                elif previous["sector_name"] != sym["sector"] or previous["industry_name"] != sym.get("industry", ""):
                    db.execute(text("""
                        UPDATE universe_sector_history SET effective_to = :effective_from
                        WHERE ticker_symbol = :ticker AND effective_to IS NULL
                    """), {"ticker": sym["symbol"], "effective_from": sync_date})
                    db.execute(text("""
                        INSERT INTO universe_sector_history
                            (ticker_symbol, sector_name, industry_name, effective_from)
                        VALUES (:ticker, :sector, :industry, :effective_from)
                    """), {"ticker": sym["symbol"], "sector": sym["sector"],
                           "industry": sym.get("industry", ""), "effective_from": sync_date})
                synced_count += 1

        logger.success(f"Universe sync complete: {synced_count} stocks active")

    except NotImplementedError as e:
        logger.warning(f"Universe sync skipped (broker auth not implemented): {e}")
    except Exception as e:
        logger.error(f"Universe sync FAILED: {e}")
        raise
    finally:
        client.close()

    return synced_count


if __name__ == "__main__":
    sync_nifty100_universe()
