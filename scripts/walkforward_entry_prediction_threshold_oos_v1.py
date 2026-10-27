#!/usr/bin/env python3
"""Validate raw probability thresholds with walk-forward OOS selection.

Diagnostic only. A threshold is selected using only prior chronological data,
then evaluated on the next untouched block. This does not retrain, calibrate,
promote, change EntryGate, or place orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_LEDGER = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
DEFAULT_OUTPUT = Path("data/prediction/trend_pullback_threshold_walkforward_oos_v1.json")
DEFAULT_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35)
DEFAULT_FOLDS = 5
MIN_TRAIN_SELECTED = 20
MIN_TRAIN_COVERAGE_PCT = 5.0
MIN_TRAIN_LIFT = 1.10


def metrics(frame: pd.DataFrame, threshold: float) -> dict:
    baseline = float(frame["label_target_before_stop"].mean())
    selected = frame[frame["probability_target_before_stop"] >= threshold]
    n = len(selected)
    return {
        "threshold": threshold,
        "rows": int(len(frame)),
        "selected_rows": int(n),
        "coverage_pct": float(n / len(frame) * 100.0) if len(frame) else 0.0,
        "baseline_success_rate": baseline,
        "success_rate": float(selected["label_target_before_stop"].mean()) if n else None,
        "lift": float(selected["label_target_before_stop"].mean() / baseline) if n and baseline > 0 else None,
        "mean_probability": float(selected["probability_target_before_stop"].mean()) if n else None,
    }


def choose_threshold(train: pd.DataFrame, thresholds: list[float]) -> tuple[float, list[dict]]:
    candidates = [metrics(train, t) for t in thresholds]
    eligible = [
        row for row in candidates
        if row["selected_rows"] >= MIN_TRAIN_SELECTED
        and row["coverage_pct"] >= MIN_TRAIN_COVERAGE_PCT
        and row["lift"] is not None
        and row["lift"] >= MIN_TRAIN_LIFT
    ]
    # Prefer the highest threshold that remains adequately populated and lifts
    # success rate. If several qualify, choose the highest threshold to favor
    # selectivity; the untouched holdout determines whether that generalizes.
    pool = eligible if eligible else candidates
    chosen = max(pool, key=lambda row: (row["threshold"], row["lift"] or -1.0))
    return float(chosen["threshold"]), candidates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.folds < 3:
        raise ValueError("--folds must be >= 3")
    thresholds = sorted(set(args.thresholds))
    if any(t < 0.0 or t > 1.0 for t in thresholds):
        raise ValueError("thresholds must be between 0 and 1")
    if not args.ledger.exists():
        raise FileNotFoundError(f"Ledger not found: {args.ledger}")

    frame = pd.read_csv(args.ledger)
    required = {"ticker_symbol", "trade_date", "setup_type", "probability_target_before_stop", "label_target_before_stop"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Ledger missing columns: {sorted(missing)}")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["probability_target_before_stop"] = pd.to_numeric(frame["probability_target_before_stop"], errors="coerce")
    frame["label_target_before_stop"] = pd.to_numeric(frame["label_target_before_stop"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "probability_target_before_stop", "label_target_before_stop"]).copy()
    frame["label_target_before_stop"] = frame["label_target_before_stop"].astype(int)
    frame = frame[frame["label_target_before_stop"].isin([0, 1])].sort_values("trade_date").reset_index(drop=True)

    blocks = np.array_split(frame, args.folds)
    fold_results = []
    oos_rows = []

    print("=== GANESHA V1 WALK-FORWARD THRESHOLD OOS V1 ===")
    print(f"ledger={args.ledger}")
    print(f"rows={len(frame)} | tickers={frame['ticker_symbol'].nunique()} | folds={args.folds}")
    print(f"thresholds={','.join(f'{t:.2f}' for t in thresholds)}")

    for idx in range(1, len(blocks)):
        train = pd.concat(blocks[:idx], ignore_index=True)
        holdout = blocks[idx].copy()
        chosen, train_candidates = choose_threshold(train, thresholds)
        holdout_metrics = metrics(holdout, chosen)
        holdout_metrics["fold"] = idx
        holdout_metrics["train_rows"] = int(len(train))
        holdout_metrics["train_date_end"] = train["trade_date"].max().date().isoformat()
        holdout_metrics["holdout_date_start"] = holdout["trade_date"].min().date().isoformat()
        holdout_metrics["holdout_date_end"] = holdout["trade_date"].max().date().isoformat()
        holdout_metrics["selected_threshold"] = chosen
        holdout_metrics["train_candidates"] = train_candidates
        fold_results.append(holdout_metrics)
        selected = holdout[holdout["probability_target_before_stop"] >= chosen].copy()
        if not selected.empty:
            oos_rows.append(selected)
        print(
            f"FOLD {idx} | train={len(train)} | holdout={len(holdout)}"
            f" | selected_threshold={chosen:.2f}"
            f" | holdout_selected={len(selected)}"
            f" | coverage={holdout_metrics['coverage_pct']:.2f}%"
            f" | success={holdout_metrics['success_rate'] if holdout_metrics['success_rate'] is not None else 'NA'}"
            f" | lift={holdout_metrics['lift'] if holdout_metrics['lift'] is not None else 'NA'}"
            f" | {holdout_metrics['holdout_date_start']} -> {holdout_metrics['holdout_date_end']}"
        )

    if oos_rows:
        combined_selected = pd.concat(oos_rows, ignore_index=True)
        combined_holdout = pd.concat(blocks[1:], ignore_index=True)
        combined = metrics(combined_holdout, 0.0)
        combined["selected_rows"] = int(len(combined_selected))
        combined["coverage_pct"] = float(len(combined_selected) / len(combined_holdout) * 100.0)
        combined["success_rate"] = float(combined_selected["label_target_before_stop"].mean())
        combined["lift"] = float(combined["success_rate"] / combined["baseline_success_rate"]) if combined["baseline_success_rate"] > 0 else None
        combined["mean_probability"] = float(combined_selected["probability_target_before_stop"].mean())
    else:
        combined = {"selected_rows": 0, "coverage_pct": 0.0, "success_rate": None, "lift": None}

    print("\n=== COMBINED UNTOUCHED OOS ===")
    print(json.dumps(combined, indent=2))

    chosen_values = [row["selected_threshold"] for row in fold_results]
    result = {
        "experiment": "trend_pullback_raw_probability_threshold_walkforward_oos_v1",
        "decision": "diagnostic_only",
        "production_ready": False,
        "promotion_status": "candidate_only",
        "entry_gate_changed": False,
        "calibration_deployed": False,
        "ledger": str(args.ledger),
        "rows": int(len(frame)),
        "tickers": int(frame["ticker_symbol"].nunique()),
        "date_range": [frame["trade_date"].min().date().isoformat(), frame["trade_date"].max().date().isoformat()],
        "thresholds": thresholds,
        "selection_policy": {
            "min_train_selected": MIN_TRAIN_SELECTED,
            "min_train_coverage_pct": MIN_TRAIN_COVERAGE_PCT,
            "min_train_lift": MIN_TRAIN_LIFT,
            "selection_rule": "highest threshold meeting minimum training population and lift constraints",
        },
        "folds": fold_results,
        "combined_untouched_oos": combined,
        "selected_thresholds": chosen_values,
        "interpretation": "Thresholds are diagnostic candidate-selection levels only; OOS results do not authorize EntryGate changes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"SAVED: {args.output}")
    print("DECISION: diagnostic only; no production promotion or EntryGate changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
