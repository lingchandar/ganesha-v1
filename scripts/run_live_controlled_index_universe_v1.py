"""Run the read-only live monitor on the current NIFTY50 + BANKNIFTY universe.

The universe is fetched from the project's official NSE Indices loader at runtime,
so it follows the current controlled index membership instead of hard-coding a
59-stock assumption. No orders are placed by this runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from src.ingestion.live_swing_monitor import LiveSwingMonitor
from src.ingestion.nse_index_universe import load_controlled_index_universe

DEFAULT_SHADOW_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-model", type=Path, default=DEFAULT_SHADOW_MODEL)
    args = parser.parse_args(argv)

    symbols = load_controlled_index_universe()
    print(f"=== GANESHA V1 CONTROLLED LIVE UNIVERSE ===")
    print("SOURCE: current NIFTY50 + BANKNIFTY constituents")
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
        print("\n=== CONTROLLED LIVE MONITOR STOPPED ===", flush=True)
    finally:
        monitor.candle_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
