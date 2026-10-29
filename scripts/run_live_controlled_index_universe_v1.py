"""Run the read-only live monitor on the versioned 50-stock Trend Pullback universe.

This is the live counterpart of the historical population validator. It uses
exactly the same fixed symbols as the reproducible Trend Pullback research
universe, so historical and live tests are population-aligned.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from src.ingestion.live_swing_monitor import LiveSwingMonitor
from scripts.trend_pullback_validation_universe_v1 import TREND_PULLBACK_VALIDATION_SYMBOLS

DEFAULT_SHADOW_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-model", type=Path, default=DEFAULT_SHADOW_MODEL)
    args = parser.parse_args(argv)

    symbols = tuple(TREND_PULLBACK_VALIDATION_SYMBOLS)
    print("=== GANESHA V1 TREND PULLBACK LIVE UNIVERSE ===")
    print("SOURCE: versioned TREND_PULLBACK_VALIDATION_SYMBOLS")
    print(f"UNIQUE SYMBOLS: {len(symbols)}")
    print("SYMBOLS:")
    print(" ".join(symbols))
    print("MODE: read-only live monitoring; no order placement")

    monitor = LiveSwingMonitor(symbols, shadow_model_path=args.shadow_model)
    print(f"SHADOW MODEL AVAILABLE: {monitor.shadow_predictor.available}", flush=True)
    try:
        counts = monitor.run()
        print(f"WARM-UP COMPLETE: {counts}", flush=True)
    except KeyboardInterrupt:
        print("\n=== TREND PULLBACK LIVE MONITOR STOPPED ===", flush=True)
    finally:
        monitor.candle_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
