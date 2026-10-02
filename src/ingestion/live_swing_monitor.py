"""Startup orchestration for Ganesha's read-only live swing monitor."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

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

    def evaluate_daily_exits(
        self,
        evaluation_date: date,
        client: Optional[MarketDataClient] = None,
    ) -> dict[str, object]:
        """Evaluate open trades using one completed daily candle per symbol.

        This is intentionally separate from the 15-minute entry stream because
        PrecisionExitEngine is a daily-bar exit engine. The supplied date must
        represent a completed trading session.
        """
        manager = self.analysis.trade_manager
        if not manager.active_trades:
            return {}

        owns_client = client is None
        market_client = client or MarketDataClient()
        try:
            if owns_client and not market_client.authenticate():
                raise RuntimeError("FYERS authentication is unavailable.")

            results = {}
            for symbol in list(manager.active_trades):
                history = market_client.fetch_historical_candles(
                    symbol,
                    evaluation_date,
                    evaluation_date,
                    resolution="D",
                )
                if history.empty:
                    continue

                row = history.iloc[-1]
                daily_candle = {
                    "open": float(row["open_price"]),
                    "high": float(row["high_price"]),
                    "low": float(row["low_price"]),
                    "close": float(row["close_price"]),
                }
                results[symbol] = manager.evaluate_daily_candle(
                    ticker_symbol=symbol,
                    evaluation_date=evaluation_date,
                    daily_candle=daily_candle,
                )
            return results
        finally:
            if owns_client:
                market_client.close()

    def start(self) -> None:
        """Start the read-only live WebSocket after warm-up."""
        self.candle_engine.start(self.symbols)

    def run(self, lookback_days: int = 7) -> dict[str, int]:
        """Warm up and then block on the live WebSocket."""
        counts = self.warm_up(lookback_days)
        self.start()
        self.candle_engine.run_forever()
        return counts
