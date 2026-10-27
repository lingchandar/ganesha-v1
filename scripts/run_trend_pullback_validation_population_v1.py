"""Run the candidate model's historical Trend Pullback validation on a fixed 50-stock universe."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from scripts.trend_pullback_validation_universe_v1 import TREND_PULLBACK_VALIDATION_SYMBOLS
from scripts.evaluate_entry_prediction_historical_population_v1 import evaluate_symbol, _load_model, summarize

DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
DEFAULT_OUTPUT = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
DEFAULT_SESSIONS = 360


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.sessions < 1:
        raise ValueError("--sessions must be >= 1")

    bundle = _load_model(args.model)
    print(
        "=== GANESHA V1 TREND PULLBACK POPULATION VALIDATION ===\n"
        f"universe={len(TREND_PULLBACK_VALIDATION_SYMBOLS)} stocks | sessions={args.sessions} | "
        f"model={args.model}"
    )

    rows = []
    for symbol in TREND_PULLBACK_VALIDATION_SYMBOLS:
        try:
            rows.extend(evaluate_symbol(symbol, args.sessions, bundle, "TREND_PULLBACK"))
        except Exception as exc:
            print(f"POPULATION ERROR | {symbol} | {type(exc).__name__}: {exc}")

    import pandas as pd

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"TREND PULLBACK LEDGER: {args.output} | rows={len(frame)}")
    else:
        print("TREND PULLBACK LEDGER: no evaluable candidates")

    summarize(frame, "TREND_PULLBACK")
    if len(frame) < 500:
        print(
            f"NEXT GATE: population still below 500 candidates ({len(frame)}); "
            "do not retrain or promote."
        )
    else:
        print("NEXT GATE: population >= 500; proceed to stability/calibration analysis, not promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
