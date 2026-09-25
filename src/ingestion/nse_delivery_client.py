"""
GANESHA V1 — NSE Official Delivery Data Engine (Module 7)

SEBI & INSTITUTIONAL REALISM:
Broker OHLCV candles do not reliably include exchange delivery breakdown.
This module ingests official NSE delivery position data directly from NSE India archives
(Security-wise Delivery Position Bhavcopy - sec_bhavdata_full_DDMMYYYY.csv or MTO).

Key Metrics Computed:
- Delivery Volume (DELIV_QTY)
- Delivery Percentage (DELIV_PER)
- 5-Day Delivery Volume SMA (Trailing 5 sessions)
- Delivery Expansion Ratio = Current Delivery Volume / 5-Day Delivery SMA
- Institutional Accumulation Gate: Ratio > 1.5x (or > 1.725x during monthly expiry week)
"""
import io
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import httpx
import pandas as pd
from loguru import logger

from src.ingestion.execution_shield import intercept_outbound_request, throttle


class NSEArchiveDeliveryClient:
    """
    Ingests and parses official NSE end-of-day delivery volumes and delivery percentages.
    All outbound requests pass through the SEBI compliance shield and rate limiter.
    """

    BASE_URL = "https://archives.nseindia.com/products/content"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    def __init__(self, timeout: float = 30.0):
        self._session = httpx.Client(timeout=timeout, headers=self.HEADERS, follow_redirects=True)

    def fetch_daily_delivery_bhavcopy(self, trade_date: date) -> pd.DataFrame:
        """
        Download and parse the full NSE security-wise delivery bhavcopy for a specific date.
        Filename format: sec_bhavdata_full_DDMMYYYY.csv

        Returns:
            DataFrame with normalized columns:
            [ticker_symbol, series, open, high, low, close, volume_traded, delivery_volume, delivery_percentage]
        """
        date_str = trade_date.strftime("%d%m%Y")
        url = f"{self.BASE_URL}/sec_bhavdata_full_{date_str}.csv"

        # Intercept for SEBI compliance
        intercept_outbound_request(url)
        throttle.wait_for_slot()

        logger.info(f"Fetching NSE delivery bhavcopy for {trade_date.isoformat()} from: {url}")

        try:
            response = self._session.get(url)
            if response.status_code == 404:
                logger.warning(f"No delivery bhavcopy available for {trade_date} (likely holiday or weekend).")
                return pd.DataFrame()
            response.raise_for_status()

            # Read CSV content
            df = pd.read_csv(io.StringIO(response.text))

            # Strip whitespace from column names and string cells
            df.columns = [c.strip() for c in df.columns]
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype(str).str.strip()

            # Filter for Equity series ('EQ')
            if "SERIES" in df.columns:
                df = df[df["SERIES"] == "EQ"].copy()

            # Clean and normalize columns
            # Raw columns: SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE,
            # LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS,
            # NO_OF_TRADES, DELIV_QTY, DELIV_PER
            rename_map = {
                "SYMBOL": "ticker_symbol",
                "OPEN_PRICE": "open_price",
                "HIGH_PRICE": "high_price",
                "LOW_PRICE": "low_price",
                "CLOSE_PRICE": "close_price",
                "TTL_TRD_QNTY": "volume_traded",
                "DELIV_QTY": "delivery_volume",
                "DELIV_PER": "delivery_percentage",
            }
            df = df.rename(columns=rename_map)

            # Convert numeric columns
            numeric_cols = [
                "open_price", "high_price", "low_price", "close_price",
                "volume_traded", "delivery_volume", "delivery_percentage"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].replace("-", 0), errors="coerce").fillna(0)

            df["trade_date"] = trade_date
            logger.info(f"Successfully parsed {len(df)} EQ records for {trade_date.isoformat()}")
            return df

        except Exception as e:
            logger.error(f"Error fetching NSE delivery data for {trade_date}: {e}")
            return pd.DataFrame()

    def calculate_delivery_expansion(
        self,
        recent_delivery_history: List[int],
        today_delivery: int,
        is_expiry_week: bool = False
    ) -> Dict[str, float]:
        """
        Module 7 & 16a Formulation:
        - Delivery 5-Day SMA = mean(past 5 trading sessions delivery)
        - Delivery Ratio = today_delivery / delivery_5d_sma
        - Normal Threshold: 1.50x
        - Expiry-Week Threshold: 1.50 * 1.15 = 1.725x

        Args:
            recent_delivery_history: List of delivery volumes for past 5 sessions [t-5, t-4, t-3, t-2, t-1]
            today_delivery: Current session delivery volume (t)
            is_expiry_week: True if within final 3 trading sessions of monthly F&O expiry

        Returns:
            Dict containing metrics and pass/fail gate status.
        """
        if len(recent_delivery_history) < 5:
            return {
                "delivery_5d_sma": 0.0,
                "expansion_ratio": 0.0,
                "threshold_required": 1.725 if is_expiry_week else 1.50,
                "is_institutional_accumulation": False,
                "rejection_reason": "INSUFFICIENT_DELIVERY_HISTORY_LESS_THAN_5_DAYS"
            }

        delivery_sma = sum(recent_delivery_history[-5:]) / 5.0
        expansion_ratio = today_delivery / delivery_sma if delivery_sma > 0 else 0.0
        threshold = 1.725 if is_expiry_week else 1.50

        passes_gate = expansion_ratio >= threshold

        return {
            "today_delivery": float(today_delivery),
            "delivery_5d_sma": round(delivery_sma, 2),
            "expansion_ratio": round(expansion_ratio, 3),
            "threshold_required": threshold,
            "is_institutional_accumulation": passes_gate,
            "rejection_reason": None if passes_gate else (
                f"DELIVERY_EXPANSION_INSUFFICIENT: {expansion_ratio:.2f}x < {threshold:.3f}x SMA"
            )
        }

    def close(self):
        self._session.close()
