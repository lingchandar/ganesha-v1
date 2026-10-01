"""Aggregate completed OHLCV candles into a higher analysis timeframe."""

from __future__ import annotations

from typing import Literal

import pandas as pd


class TimeframeAggregator:
    """Deterministically aggregate lower-timeframe candles."""

    def __init__(self, timeframe_minutes: int = 15) -> None:
        if timeframe_minutes < 1:
            raise ValueError("timeframe_minutes must be >= 1")
        self.timeframe_minutes = timeframe_minutes

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        required = {
            "timestamp",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume_traded",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = (
            data.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .set_index("timestamp")
        )

        rule = f"{self.timeframe_minutes}min"
        result = data.resample(rule, label="left", closed="left").agg(
            open_price=("open_price", "first"),
            high_price=("high_price", "max"),
            low_price=("low_price", "min"),
            close_price=("close_price", "last"),
            volume_traded=("volume_traded", "sum"),
        )

        counts = data["close_price"].resample(rule, label="left", closed="left").count()
        result["source_candle_count"] = counts
        result = result.dropna(subset=["open_price", "high_price", "low_price", "close_price"])
        result = result.reset_index()

        # Only completed higher-timeframe bars are valid for analysis.
        result = result[result["source_candle_count"] == self.timeframe_minutes].copy()
        return result.reset_index(drop=True)
