"""Startup orchestration for Ganesha's read-only live swing monitor."""

from __future__ import annotations

from typing import Iterable

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.live_candle_engine import LiveCandleEngine
from src.ingestion.market_data_client import MarketDataClient


class LiveSwingMonitor:
    """Warm symbols from FYERS history and then stream live ticks."""

    def __init__(self, symbols: Iterable[str], signal_handler=None) -> None:
        self.symbols = tuple(sorted({str(s).strip() for s in symbols if str(s).strip()}))
        if not self.symbols:
            raise ValueError("At least one FYERS symbol is required")

        self.analysis = LiveAnalysisEngine(signal_handler=signal_handler)
        self.candle_engine = LiveCandleEngine(candle_handler=self.analysis.on_candle)

    def warm_up(self, lookback_days: int = 7) -> dict[str, int]:
        """Load recent completed 15-minute history for every subscribed symbol."""
        counts: dict[str, int] = {}
        with MarketDataClient() as client:
            if not client.authenticate():
                raise RuntimeError("FYERS authentication is unavailable.")
            for symbol in self.symbols:
                counts[symbol] = self.analysis.warm_up(symbol, client, lookback_days)
        return counts

    def start(self) -> None:
        """Start the read-only live WebSocket after warm-up."""
        self.candle_engine.start(self.symbols)

    def run(self, lookback_days: int = 7) -> dict[str, int]:
        """Warm up and then block on the live WebSocket."""
        counts = self.warm_up(lookback_days)
        self.start()
        self.candle_engine.run_forever()
        return counts
