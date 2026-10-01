"""Rolling in-memory OHLCV buffer for live Ganesha analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from threading import RLock
from typing import Dict, Optional

import pandas as pd

from src.ingestion.one_minute_candle_builder import OneMinuteCandle


class LiveOHLCVBuffer:
    """Keep a bounded candle history independently for each symbol."""

    COLUMNS = [
        "timestamp",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume_traded",
    ]

    def __init__(self, max_candles: int = 300) -> None:
        if max_candles < 1:
            raise ValueError("max_candles must be >= 1")
        self.max_candles = max_candles
        self._data: Dict[str, deque[OneMinuteCandle]] = defaultdict(
            lambda: deque(maxlen=self.max_candles)
        )
        self._lock = RLock()

    def add(self, candle: OneMinuteCandle) -> None:
        with self._lock:
            candles = self._data[candle.symbol]
            if candles and candle.timestamp <= candles[-1].timestamp:
                return
            candles.append(candle)

    def get(self, symbol: str) -> pd.DataFrame:
        with self._lock:
            rows = list(self._data.get(symbol, ()))
        return pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open_price": c.open_price,
                    "high_price": c.high_price,
                    "low_price": c.low_price,
                    "close_price": c.close_price,
                    "volume_traded": c.volume_traded,
                }
                for c in rows
            ],
            columns=self.COLUMNS,
        )

    def count(self, symbol: str) -> int:
        with self._lock:
            return len(self._data.get(symbol, ()))

    def clear(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol is None:
                self._data.clear()
            else:
                self._data.pop(symbol, None)
