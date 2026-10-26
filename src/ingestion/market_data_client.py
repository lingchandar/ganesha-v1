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
from datetime import date, datetime, timezone, timedelta
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
    MAX_DAILY_HISTORY_DAYS = 366

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
        Fetch historical OHLCV candle data from Fyers API v3.

        FYERS limits daily/weekly/monthly history requests to 366 calendar days
        per request, so longer ranges are fetched in sequential chunks and then
        combined. This keeps the caller's requested range intact.

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
        if from_date > to_date:
            raise ValueError("from_date must be <= to_date")

        url = f"{self.FYERS_DATA_URL}/history"
        frames: list[pd.DataFrame] = []
        chunk_start = from_date

        try:
            while chunk_start <= to_date:
                chunk_end = min(
                    chunk_start + timedelta(days=self.MAX_DAILY_HISTORY_DAYS - 1),
                    to_date,
                )
                params = {
                    "symbol": symbol,
                    "resolution": resolution,
                    "date_format": "1",  # YYYY-MM-DD
                    "range_from": chunk_start.strftime("%Y-%m-%d"),
                    "range_to": chunk_end.strftime("%Y-%m-%d"),
                    "cont_flag": "1",
                }

                resp = self._safe_request("GET", url, params=params)
                data = resp.json()

                if data.get("s") != "ok" or "candles" not in data:
                    logger.warning(
                        f"No candles returned for {symbol} ({chunk_start} to {chunk_end}): "
                        f"{data.get('message', 'Unknown response')}"
                    )
                else:
                    raw_candles = data["candles"]
                    if raw_candles:
                        frame = pd.DataFrame(
                            raw_candles,
                            columns=[
                                "epoch_time",
                                "open_price",
                                "high_price",
                                "low_price",
                                "close_price",
                                "volume_traded",
                            ],
                        )
                        frame["timestamp"] = (
                            pd.to_datetime(frame["epoch_time"], unit="s", utc=True)
                            .dt.tz_convert("Asia/Kolkata")
                        )
                        frame = frame.drop(columns=["epoch_time"])
                        frames.append(frame)
                        logger.info(
                            f"Fetched {len(frame)} candles for {symbol} "
                            f"({chunk_start} to {chunk_end})"
                        )

                chunk_start = chunk_end + timedelta(days=1)

            if not frames:
                return pd.DataFrame()

            df = pd.concat(frames, ignore_index=True)
            df = (
                df.sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )
            logger.info(f"Fetched {len(df)} candles total for {symbol} ({from_date} to {to_date})")
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
