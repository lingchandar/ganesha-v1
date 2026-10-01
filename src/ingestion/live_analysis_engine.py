"""Live candle -> analysis pipeline for Ganesha swing monitoring."""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from src.ingestion.live_ohlcv_buffer import LiveOHLCVBuffer
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.setups.swing_setups import SwingSetupScanner


class LiveAnalysisEngine:
    """Maintain live candles and run deterministic setup scans on completed bars.

    The scanner is invoked only after a completed candle arrives. The buffer
    intentionally requires a configurable warm-up before analysis.
    """

    def __init__(
        self,
        min_candles: int = 60,
        max_candles: int = 300,
        scanner: Optional[SwingSetupScanner] = None,
        signal_handler: Optional[Callable[[str, list], None]] = None,
    ) -> None:
        if min_candles < 60:
            raise ValueError("min_candles must be >= 60 for the swing scanner")
        self.min_candles = min_candles
        self.buffer = LiveOHLCVBuffer(max_candles=max_candles)
        self.scanner = scanner or SwingSetupScanner()
        self.signal_handler = signal_handler

    def on_candle(self, candle: OneMinuteCandle) -> list:
        """Store a completed candle and scan once warm-up is available."""
        self.buffer.add(candle)
        if self.buffer.count(candle.symbol) < self.min_candles:
            return []

        df = self.buffer.get(candle.symbol)
        signals = self.scanner.scan_all_setups(
            ticker_symbol=candle.symbol,
            df=df,
        )
        if self.signal_handler:
            self.signal_handler(candle.symbol, signals)
        return signals

    def dataframe(self, symbol: str) -> pd.DataFrame:
        return self.buffer.get(symbol)
