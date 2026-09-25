"""
GANESHA V1 — Read-Only Market Data Client (Module 1, 2)

SEBI COMPLIANCE & INSTITUTIONAL REALISM:
This client wraps the Fyers API v3 for fetching historical daily OHLCV candles
and live market quotes exclusively.

STRUCTURAL GUARANTEE:
1. No order execution methods (place_order, modify_order, cancel_order) exist in this class.
2. Every outbound request passes through `intercept_outbound_request()`, which terminates
   the process with a fatal RuntimeError if any order endpoint is targeted.
3. Every request passes through `throttle.wait_for_slot()` (250ms throttle, 120 req/min).
"""
import httpx
import time
from datetime import date, datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from src.core.config import settings
from src.ingestion.execution_shield import intercept_outbound_request, throttle
from src.ingestion.fyers_auth import FyersAuthManager


class MarketDataClient:
    """
    Read-only market data client for Fyers API v3.
    """

    FYERS_DATA_URL = "https://api-t1.fyers.in/data"

    def __init__(self):
        self.broker = settings.broker_name.lower()
        self._session = httpx.Client(timeout=30.0)
        self._auth_token: Optional[str] = None
        self.auth_manager = FyersAuthManager()
        logger.info(f"MarketDataClient initialized for broker: {self.broker}")

    def _safe_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Inspect URL through SEBI compliance interceptor and apply rate limit throttle.
        """
        intercept_outbound_request(url)
        throttle.wait_for_slot()

        # Inject Authorization header if token exists
        if self._auth_token:
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"{settings.fyers_app_id}:{self._auth_token}"
            kwargs["headers"] = headers

        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def authenticate(self) -> bool:
        """
        Authenticate with Fyers API by checking for valid cached token
        or prompting user.
        """
        token = self.auth_manager.load_cached_token()
        if token:
            self._auth_token = token
            logger.info("Fyers daily access token loaded from cache.")
            return True
        else:
            logger.warning(
                "No valid Fyers token found for today's session. "
                "Run '.venv/bin/python -m src.ingestion.fyers_auth' once your API key is ready."
            )
            return False

    def fetch_historical_candles(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        resolution: str = "D",
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV candle data from Fyers API v3.

        Args:
            symbol: Fyers format symbol, e.g. "NSE:RELIANCE-EQ" or "NSE:NIFTY50-INDEX"
            from_date: Start date (inclusive)
            to_date: End date (inclusive)
            resolution: "D" for daily

        Returns:
            DataFrame with [timestamp, open_price, high_price, low_price, close_price, volume_traded]
        """
        if not self._auth_token:
            raise RuntimeError(
                "MarketDataClient is unauthenticated. Call authenticate() with a valid token first."
            )

        url = f"{self.FYERS_DATA_URL}/history"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",  # YYYY-MM-DD
            "range_from": from_date.strftime("%Y-%m-%d"),
            "range_to": to_date.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        try:
            resp = self._safe_request("GET", url, params=params)
            data = resp.json()

            if data.get("s") != "ok" or "candles" not in data:
                logger.warning(f"No candles returned for {symbol}: {data.get('message', 'Unknown response')}")
                return pd.DataFrame()

            raw_candles = data["candles"]
            # Fyers candle structure: [epoch_sec, open, high, low, close, volume]
            df = pd.DataFrame(
                raw_candles,
                columns=["epoch_time", "open_price", "high_price", "low_price", "close_price", "volume_traded"]
            )
            df["timestamp"] = pd.to_datetime(df["epoch_time"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
            df = df.drop(columns=["epoch_time"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} candles for {symbol} ({from_date} to {to_date})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical candles for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch quotes for multiple symbols.
        """
        if not self._auth_token:
            raise RuntimeError("MarketDataClient unauthenticated.")

        url = f"{self.FYERS_DATA_URL}/quotes"
        params = {"symbols": ",".join(symbols)}

        resp = self._safe_request("GET", url, params=params)
        data = resp.json()

        results = {}
        if data.get("s") == "ok" and "d" in data:
            for item in data["d"]:
                val = item.get("v", {})
                sym = item.get("n", "")
                results[sym] = {
                    "symbol": sym,
                    "ltp": val.get("lp"),
                    "open": val.get("open_price"),
                    "high": val.get("high_price"),
                    "low": val.get("low_price"),
                    "close": val.get("prev_close_price"),
                    "volume": val.get("volume"),
                }
        return results

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
