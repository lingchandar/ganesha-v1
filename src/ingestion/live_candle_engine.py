"""Live FYERS tick-to-candle engine.

This is the first runtime bridge between the real-time FYERS WebSocket and
Ganesha's OHLCV-based analysis pipeline. It remains read-only.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from src.ingestion.fyers_realtime import FyersRealtimeClient, LiveMarketTick
from src.ingestion.one_minute_candle_builder import OneMinuteCandle, OneMinuteCandleBuilder


class LiveCandleEngine:
    """Connect FYERS ticks to deterministic one-minute OHLCV candles."""

    def __init__(
        self,
        candle_handler: Optional[Callable[[OneMinuteCandle], None]] = None,
        error_handler=None,
        close_handler=None,
    ) -> None:
        self.candle_builder = OneMinuteCandleBuilder(candle_handler)
        self.realtime = FyersRealtimeClient(
            tick_handler=self._on_tick,
            error_handler=error_handler,
            close_handler=close_handler,
        )

    def _on_tick(self, tick: LiveMarketTick) -> None:
        self.candle_builder.update(tick)

    def start(self, symbols: Iterable[str]) -> None:
        """Start the live read-only stream for the requested symbols."""
        self.realtime.connect(symbols)

    def stop(self) -> list[OneMinuteCandle]:
        """Flush open candles before stopping the application."""
        return self.candle_builder.flush()

    def run_forever(self) -> None:
        self.realtime.keep_running()
