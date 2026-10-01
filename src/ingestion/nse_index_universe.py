"""Current NSE index constituent loader for Ganesha."""

from src.ingestion.nse_security_master import to_fyers_equity_symbol

NSE_INDEX_API_URL = "https://www.nseindia.com/api/equity-stockIndices"
INDEX_NAMES = {"NIFTY50": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}

def normalize_index_symbols(symbols):
    return tuple(sorted({to_fyers_equity_symbol(s) for s in symbols if str(s).strip()}))

def fetch_index_constituents(index, session=None, timeout=30):
    import requests
    key = str(index).strip().upper().replace(" ", "")
    if key not in INDEX_NAMES:
        raise ValueError(f"Unsupported NSE index: {index}")
    http = session or requests.Session()
    http.headers.update({"User-Agent": "Mozilla/5.0 GaneshaV1/1.0", "Accept": "application/json", "Referer": "https://www.nseindia.com/"})
    response = http.get(NSE_INDEX_API_URL, params={"index": INDEX_NAMES[key]}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("NSE index response does not contain a data list")
    result = normalize_index_symbols(row.get("symbol") for row in rows if isinstance(row, dict) and row.get("symbol"))
    if not result:
        raise ValueError(f"NSE returned no constituents for {INDEX_NAMES[key]}")
    return result

def load_controlled_index_universe(session=None, timeout=30):
    symbols = set()
    for index in ("NIFTY50", "BANKNIFTY"):
        symbols.update(fetch_index_constituents(index, session=session, timeout=timeout))
    return tuple(sorted(symbols))
