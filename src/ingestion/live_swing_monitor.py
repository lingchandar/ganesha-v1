"""Startup orchestration for Ganesha's read-only live swing monitor."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.live_candle_engine import LiveCandleEngine
from src.ingestion.market_data_client import MarketDataClient
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.prediction.candidate_shadow import CandidateShadowPredictor


class LiveSwingMonitor:
    """Warm symbols from FYERS history and then stream live ticks."""

    def __init__(
        self,
        symbols: Iterable[str],
        signal_handler: Optional[Callable[[str, list], None]] = None,
        shadow_model_path: Path | str = "models/entry_prediction_extra_trees_candidate_v1.joblib",
    ) -> None:
        self.symbols = tuple(sorted({str(s).strip() for s in symbols if str(s).strip()}))
        if not self.symbols:
            raise ValueError("At least one FYERS symbol is required")

        self.shadow_predictor = CandidateShadowPredictor(shadow_model_path)
        self._signal_handler = signal_handler
        self.analysis = LiveAnalysisEngine(signal_handler=self._handle_signals)
        self.candle_engine = LiveCandleEngine(candle_handler=self._handle_candle)

    def _handle_signals(self, symbol: str, signals: list) -> None:
        """Observe daily signals with the candidate without affecting decisions."""
        if self._signal_handler:
            self._signal_handler(symbol, signals)

        if not self.shadow_predictor.available:
            return

        features = self.analysis.dataframe(symbol)
        for signal in signals:
            shadow = self.shadow_predictor.predict(signal, features)
            if not shadow.available:
                continue
            print(
                "SHADOW ENTRY PREDICTION | "
                f"{shadow.symbol} | {shadow.setup_type} | "
                f"P={shadow.probability_target_before_stop:.4f} | "
                f"CONF={shadow.confidence:.4f} | "
                f"MODEL={shadow.model_version}",
                flush=True,
            )

    def _handle_candle(self, candle: OneMinuteCandle) -> None:
        """Log completed one-minute candles before forwarding them to analysis."""
        print(
            f"LIVE 1M | {candle.symbol} | {candle.timestamp.isoformat()} | "
            f"O={candle.open_price:.2f} H={candle.high_price:.2f} "
            f"L={candle.low_price:.2f} C={candle.close_price:.2f} "
            f"V={candle.volume_traded} TICKS={candle.tick_count}",
            flush=True,
        )
        self.analysis.on_candle(candle)

    def warm_up(self, lookback_days: int = 7) -> dict[str, int]:
        """Load completed daily history for every subscribed symbol."""
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

        This is intentionally separate from the daily entry stream because
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the live read-only monitor."""
    parser = argparse.ArgumentParser(description="Run Ganesha's read-only live swing monitor")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="FYERS symbols to monitor, e.g. NSE:TCS-EQ NSE:INFY-EQ",
    )
    parser.add_argument(
        "--shadow-model",
        type=Path,
        default=Path("models/entry_prediction_extra_trees_candidate_v1.joblib"),
        help="Candidate model bundle for read-only shadow inference",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the live monitor from the command line."""
    args = build_parser().parse_args(argv)
    monitor = LiveSwingMonitor(args.symbols, shadow_model_path=args.shadow_model)

    print(f"=== GANESHA V1 LIVE MONITOR | {', '.join(monitor.symbols)} ===", flush=True)
    print(f"SHADOW MODEL AVAILABLE: {monitor.shadow_predictor.available}", flush=True)
    try:
        counts = monitor.run()
        print(f"WARM-UP COMPLETE: {counts}", flush=True)
    except KeyboardInterrupt:
        print("\n=== LIVE MONITOR STOPPED ===", flush=True)
    finally:
        monitor.candle_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
