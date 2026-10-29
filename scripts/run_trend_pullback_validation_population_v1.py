"""Run resumable candidate-model Trend Pullback validation on a fixed 50-stock universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from scripts.trend_pullback_validation_universe_v1 import TREND_PULLBACK_VALIDATION_SYMBOLS
from scripts.evaluate_entry_prediction_historical_population_v1 import (
    evaluate_symbol,
    _load_model,
    summarize,
)

DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
DEFAULT_OUTPUT = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
DEFAULT_PROGRESS = Path("data/prediction/trend_pullback_population_v1_progress.json")
DEFAULT_SESSIONS = 360
SETUP_TYPE = "TREND_PULLBACK"


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {"completed_symbols": [], "failed_symbols": {}}
    return json.loads(path.read_text())


def _save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(progress, indent=2, sort_keys=True))
    tmp.replace(path)


def _save_ledger(path: Path, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)

    if args.sessions < 1:
        raise ValueError("--sessions must be >= 1")

    if args.reset:
        args.progress.unlink(missing_ok=True)
        args.output.unlink(missing_ok=True)

    bundle = _load_model(args.model)
    progress = _load_progress(args.progress)
    completed = set(progress.get("completed_symbols", []))
    failed = dict(progress.get("failed_symbols", {}))

    rows: list[dict] = []
    if args.output.exists():
        existing = pd.read_csv(args.output)
        if not existing.empty:
            rows = existing.to_dict("records")

    print(
        "=== GANESHA V1 TREND PULLBACK POPULATION VALIDATION ===\n"
        f"universe={len(TREND_PULLBACK_VALIDATION_SYMBOLS)} stocks | sessions={args.sessions} | "
        f"completed={len(completed)} | pending={len(TREND_PULLBACK_VALIDATION_SYMBOLS) - len(completed)} | "
        f"model={args.model}"
    )

    for index, symbol in enumerate(TREND_PULLBACK_VALIDATION_SYMBOLS, start=1):
        if symbol in completed:
            print(f"[{index:02d}/{len(TREND_PULLBACK_VALIDATION_SYMBOLS):02d}] SKIP {symbol} | already completed")
            continue

        print(f"[{index:02d}/{len(TREND_PULLBACK_VALIDATION_SYMBOLS):02d}] RUN {symbol}")
        try:
            symbol_rows = evaluate_symbol(symbol, args.sessions, bundle, SETUP_TYPE)
            rows.extend(symbol_rows)
            completed.add(symbol)
            failed.pop(symbol, None)
            _save_ledger(args.output, rows)
            _save_progress(
                args.progress,
                {"completed_symbols": sorted(completed), "failed_symbols": failed},
            )
            print(f"[{index:02d}] SAVED {symbol} | rows_added={len(symbol_rows)} | total_rows={len(rows)}")
        except Exception as exc:
            failed[symbol] = f"{type(exc).__name__}: {exc}"
            _save_progress(
                args.progress,
                {"completed_symbols": sorted(completed), "failed_symbols": failed},
            )
            print(f"[{index:02d}] FAILED {symbol} | {failed[symbol]}")

    frame = _save_ledger(args.output, rows)
    print(f"TREND PULLBACK LEDGER: {args.output} | rows={len(frame)}")
    print(f"PROGRESS: {args.progress} | completed={len(completed)} | failed={len(failed)}")
    if failed:
        print("FAILED SYMBOLS")
        for symbol, reason in sorted(failed.items()):
            print(f"  {symbol}: {reason}")

    summarize(frame, SETUP_TYPE)
    if len(frame) < 500:
        print(
            f"NEXT GATE: population still below 500 candidates ({len(frame)}); "
            "retry failed symbols before any retraining or promotion."
        )
    else:
        print("NEXT GATE: population >= 500; proceed to stability/calibration analysis, not promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
