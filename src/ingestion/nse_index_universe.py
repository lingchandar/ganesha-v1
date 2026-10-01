"""Current NSE index constituent loader for Ganesha.

Uses the official NSE Indices downloadable constituent CSVs instead of the
NSE web API endpoint, which may return Access Denied/404 to automated clients.
"""

from __future__ import annotations

import csv
import io

import requests

from src.ingestion.nse_security_master import to_fyers_equity_symbol


INDEX_CONSTITUENT_URLS = {
    "NIFTY50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "BANKNIFTY": "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv",
}


def normalize_index_symbols(symbols):
    return tuple(
        sorted(
            {
                to_fyers_equity_symbol(s)
                for s in symbols
                if str(s).strip()
            }
        )
    )


def _read_index_symbols(payload: bytes) -> tuple[str, ...]:
    """Parse an NSE index constituent CSV and return raw equity symbols."""
    text = payload.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        raise ValueError("NSE index CSV has no header")

    symbol_column = next(
        (
            column
            for column in reader.fieldnames
            if str(column).strip().upper() == "SYMBOL"
        ),
        None,
    )
    if symbol_column is None:
        raise ValueError(
            f"NSE index CSV has no SYMBOL column: {reader.fieldnames}"
        )

    symbols = []
    for row in reader:
        symbol = str(row.get(symbol_column) or "").strip().upper()
        if symbol:
            symbols.append(symbol)

    if not symbols:
        raise ValueError("NSE index CSV contains no constituent symbols")

    return tuple(symbols)


def fetch_index_constituents(index, session=None, timeout=30):
    """Fetch current constituents from the official NSE Indices CSV source."""
    key = str(index).strip().upper().replace(" ", "")
    if key not in INDEX_CONSTITUENT_URLS:
        raise ValueError(f"Unsupported NSE index: {index}")

    http = session or requests.Session()
    if hasattr(http, "headers"):
        http.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 Chrome/154.0 Safari/537.36"
                ),
                "Accept": "text/csv,application/octet-stream,*/*",
                "Referer": "https://www.nseindia.com/",
            }
        )

    response = http.get(INDEX_CONSTITUENT_URLS[key], timeout=timeout)
    response.raise_for_status()

    result = normalize_index_symbols(_read_index_symbols(response.content))
    if not result:
        raise ValueError(f"NSE returned no constituents for {key}")

    return result


def load_controlled_index_universe(session=None, timeout=30):
    symbols = set()
    for index in ("NIFTY50", "BANKNIFTY"):
        symbols.update(
            fetch_index_constituents(index, session=session, timeout=timeout)
        )
    return tuple(sorted(symbols))
