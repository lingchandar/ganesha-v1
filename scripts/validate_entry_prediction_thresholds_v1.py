#!/usr/bin/env python3
"""Validate raw candidate-model probability thresholds.

Diagnostic only. This script evaluates threshold behavior on the historical
Trend Pullback population ledger. It does not retrain, calibrate, promote,
change EntryGate, or place orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_LEDGER = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
DEFAULT_OUTPUT = Path("data/prediction/trend_pullback_threshold_validation_v1.json")
DEFAULT_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35)
DEFAULT_FOLDS = 5


def _threshold_metrics(frame: pd.DataFrame, threshold: float, baseline_rate: float) -> dict[str, float | int | None]:
    selected = frame[frame["probability_target_before_stop"] >= threshold]
    rows = int(len(selected))
    if rows == 0:
        return {
            "threshold": threshold,
            "selected_rows": 0,
            "coverage_pct": 0.0,
            "success_rate": None,
            "baseline_success_rate": baseline_rate,
            "lift": None,
            "mean_probability": None,
        }
    success_rate = float(selected["label_target_before_stop"].mean())
    return {
        "threshold": threshold,
        "selected_rows": rows,
        "coverage_pct": float(rows / len(frame) * 100.0),
        "success_rate": success_rate,
        "baseline_success_rate": baseline_rate,
        "lift": float(success_rate / baseline_rate) if baseline_rate > 0 else None,
        "mean_probability": float(selected["probability_target_before_stop"].mean()),
    }


def _fold_metrics(frame: pd.DataFrame, threshold: float, folds: int) -> list[dict]:
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    blocks = np.array_split(ordered, folds)
    results = []
    for idx, block in enumerate(blocks, start=1):
        baseline = float(block["label_target_before_stop"].mean())
        metrics = _threshold_metrics(block, threshold, baseline)
        metrics["fold"] = idx
        metrics["date_start"] = block["trade_date"].min().date().isoformat()
        metrics["date_end"] = block["trade_date"].max().date().isoformat()
        results.append(metrics)
    return results


def _ticker_consistency(frame: pd.DataFrame, threshold: float) -> dict[str, float | int | None]:
    rows = []
    for ticker, group in frame.groupby("ticker_symbol"):
        selected = group[group["probability_target_before_stop"] >= threshold]
        if selected.empty:
            continue
        baseline = float(group["label_target_before_stop"].mean())
        success = float(selected["label_target_before_stop"].mean())
        rows.append({"ticker": ticker, "selected_rows": len(selected), "lift": success / baseline if baseline > 0 else np.nan, "success_rate": success})
    if not rows:
        return {"tickers_with_selected_rows": 0, "median_ticker_lift": None, "positive_lift_tickers_pct": None}
    table = pd.DataFrame(rows)
    lifts = table["lift"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "tickers_with_selected_rows": int(len(table)),
        "median_ticker_lift": float(lifts.median()) if not lifts.empty else None,
        "positive_lift_tickers_pct": float((lifts > 1.0).mean() * 100.0) if not lifts.empty else None,
    }


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

    if frame.empty:
        raise RuntimeError("Ledger has no usable rows")

    baseline = float(frame["label_target_before_stop"].mean())
    print("=== GANESHA V1 RAW PROBABILITY THRESHOLD VALIDATION V1 ===")
    print(f"ledger={args.ledger}")
    print(f"rows={len(frame)} | tickers={frame['ticker_symbol'].nunique()} | setup={frame['setup_type'].nunique()}")
    print(f"date_range={frame['trade_date'].min().date()} -> {frame['trade_date'].max().date()}")
    print(f"baseline_success_rate={baseline:.6f}")
    print(f"thresholds={','.join(f'{t:.2f}' for t in thresholds)}")

    overall = []
    fold_results = {}
    ticker_results = {}
    for threshold in thresholds:
        metrics = _threshold_metrics(frame, threshold, baseline)
        overall.append(metrics)
        fold_results[f"{threshold:.2f}"] = _fold_metrics(frame, threshold, args.folds)
        ticker_results[f"{threshold:.2f}"] = _ticker_consistency(frame, threshold)
        print(
            "THRESHOLD"
            f" {threshold:.2f} | selected={metrics['selected_rows']}"
            f" | coverage={metrics['coverage_pct']:.2f}%"
            f" | success={metrics['success_rate'] if metrics['success_rate'] is not None else 'NA'}"
            f" | lift={metrics['lift'] if metrics['lift'] is not None else 'NA'}"
            f" | mean_p={metrics['mean_probability'] if metrics['mean_probability'] is not None else 'NA'}"
        )

    print("\n=== CHRONOLOGICAL FOLD STABILITY ===")
    for threshold in thresholds:
        print(f"\nP >= {threshold:.2f}")
        for row in fold_results[f"{threshold:.2f}"]:
            print(
                f"  fold_{row['fold']} | selected={row['selected_rows']}"
                f" | coverage={row['coverage_pct']:.2f}%"
                f" | success={row['success_rate'] if row['success_rate'] is not None else 'NA'}"
                f" | lift={row['lift'] if row['lift'] is not None else 'NA'}"
                f" | {row['date_start']} -> {row['date_end']}"
            )

    print("\n=== TICKER CONSISTENCY ===")
    for threshold in thresholds:
        row = ticker_results[f"{threshold:.2f}"]
        print(
            f"P >= {threshold:.2f} | tickers_selected={row['tickers_with_selected_rows']}"
            f" | median_ticker_lift={row['median_ticker_lift'] if row['median_ticker_lift'] is not None else 'NA'}"
            f" | positive_lift_tickers={row['positive_lift_tickers_pct'] if row['positive_lift_tickers_pct'] is not None else 'NA'}%"
        )

    result = {
        "experiment": "trend_pullback_raw_probability_threshold_validation_v1",
        "decision": "diagnostic_only",
        "production_ready": False,
        "promotion_status": "candidate_only",
        "entry_gate_changed": False,
        "calibration_deployed": False,
        "ledger": str(args.ledger),
        "rows": int(len(frame)),
        "tickers": int(frame["ticker_symbol"].nunique()),
        "date_range": [frame["trade_date"].min().date().isoformat(), frame["trade_date"].max().date().isoformat()],
        "baseline_success_rate": baseline,
        "thresholds": thresholds,
        "overall": overall,
        "chronological_folds": fold_results,
        "ticker_consistency": ticker_results,
        "interpretation": "Thresholds are diagnostic candidate-selection levels only; no threshold is a BUY/ENTRY rule.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nSAVED: {args.output}")
    print("DECISION: diagnostic only; candidate remains production_ready=False and candidate_only.")
    print("NEXT: choose a threshold region only after chronological and ticker stability evidence; do not modify EntryGate from this report alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
