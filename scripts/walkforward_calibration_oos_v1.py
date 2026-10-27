#!/usr/bin/env python3
"""Walk-forward probability calibration validation for the Trend Pullback candidate.

Each fold fits calibration only on earlier observations and evaluates on the
next untouched chronological block. Diagnostic only: no model, EntryGate, or
production threshold is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

DEFAULT_LEDGER = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
DEFAULT_OUTPUT = Path("data/prediction/trend_pullback_walkforward_calibration_oos_v1.json")


def ece(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def metric_block(frame: pd.DataFrame, prob_col: str, bins: int) -> dict:
    y = frame["actual_success"].to_numpy(dtype=int)
    p = frame[prob_col].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
    out = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece(y, p, bins),
    }
    if np.unique(y).size == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = None
        out["ap"] = None
    baseline = y.mean()
    for frac, name in ((0.01, "top_1pct_lift"), (0.03, "top_3pct_lift"), (0.05, "top_5pct_lift"), (0.10, "top_10pct_lift")):
        n = max(1, int(np.ceil(len(frame) * frac)))
        top_rate = y[np.argsort(-p)[:n]].mean()
        out[name] = float(top_rate / baseline) if baseline > 0 else None
    return out


def fit_platt(train: pd.DataFrame) -> LogisticRegression:
    p = train["probability_target_before_stop"].clip(1e-6, 1 - 1e-6).to_numpy()
    x = np.log(p / (1 - p)).reshape(-1, 1)
    y = train["actual_success"].to_numpy(dtype=int)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(x, y)
    return model


def apply(model: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    p = frame["probability_target_before_stop"].clip(1e-6, 1 - 1e-6).to_numpy()
    x = np.log(p / (1 - p)).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")

    frame = pd.read_csv(args.ledger)
    date_col = "date" if "date" in frame.columns else "trade_date"
    target_col = "actual_success" if "actual_success" in frame.columns else "label_target_before_stop"
    required = {date_col, "probability_target_before_stop", target_col}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["probability_target_before_stop"] = pd.to_numeric(frame["probability_target_before_stop"], errors="coerce")
    frame["actual_success"] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=["date", "probability_target_before_stop", "actual_success"]).copy()
    frame["actual_success"] = frame["actual_success"].astype(int)
    frame = frame[frame["actual_success"].isin([0, 1])].sort_values("date").reset_index(drop=True)
    if len(frame) < args.folds * 20:
        raise RuntimeError(f"Not enough rows for {args.folds}-fold walk-forward calibration: {len(frame)}")

    # Divide the timeline into equal blocks. Fold k calibrates on all blocks
    # before k and evaluates only on block k. The first block is calibration-only.
    blocks = np.array_split(frame, args.folds + 1)
    fold_results = []
    all_holdouts = []
    for fold_idx in range(1, args.folds + 1):
        train = pd.concat(blocks[:fold_idx], ignore_index=True)
        holdout = blocks[fold_idx].copy()
        if train["actual_success"].nunique() < 2 or holdout["actual_success"].nunique() < 2:
            print(f"FOLD {fold_idx}: skipped; train/holdout lacks both outcome classes")
            continue
        calibrator = fit_platt(train)
        holdout["calibrated_probability"] = apply(calibrator, holdout)
        raw = metric_block(holdout, "probability_target_before_stop", args.bins)
        calibrated = metric_block(holdout, "calibrated_probability", args.bins)
        fold = {
            "fold": fold_idx,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train_end": train["date"].max().date().isoformat(),
            "holdout_start": holdout["date"].min().date().isoformat(),
            "holdout_end": holdout["date"].max().date().isoformat(),
            "platt_intercept": float(calibrator.intercept_[0]),
            "platt_slope": float(calibrator.coef_[0][0]),
            "raw": raw,
            "calibrated": calibrated,
            "delta": {
                "brier": calibrated["brier"] - raw["brier"],
                "ece": calibrated["ece"] - raw["ece"],
                "log_loss": calibrated["log_loss"] - raw["log_loss"],
                "auc": (calibrated["auc"] - raw["auc"]) if raw["auc"] is not None else None,
                "top_10pct_lift": calibrated["top_10pct_lift"] - raw["top_10pct_lift"],
            },
        }
        fold_results.append(fold)
        all_holdouts.append(holdout)
        print(f"FOLD {fold_idx} | train={len(train)} | holdout={len(holdout)} | {fold['holdout_start']} -> {fold['holdout_end']}")
        print("  RAW:", json.dumps(raw, sort_keys=True))
        print("  CAL:", json.dumps(calibrated, sort_keys=True))
        print("  DELTA:", json.dumps(fold["delta"], sort_keys=True))

    if not fold_results:
        raise RuntimeError("No valid walk-forward folds")

    combined = pd.concat(all_holdouts, ignore_index=True)
    combined_metrics_raw = metric_block(combined, "probability_target_before_stop", args.bins)
    combined_metrics_cal = metric_block(combined, "calibrated_probability", args.bins)
    result = {
        "experiment": "trend_pullback_walkforward_calibration_oos_v1",
        "decision": "diagnostic_only",
        "source_schema": {"date_column": date_col, "target_column": target_col},
        "rows": int(len(frame)),
        "folds_requested": args.folds,
        "folds_evaluated": len(fold_results),
        "date_range": [frame["date"].min().date().isoformat(), frame["date"].max().date().isoformat()],
        "folds": fold_results,
        "combined_untouched_holdouts": {
            "raw": combined_metrics_raw,
            "calibrated": combined_metrics_cal,
        },
        "production_ready": False,
        "promotion_status": "candidate_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== COMBINED WALK-FORWARD HOLDOUTS ===")
    print("RAW:", json.dumps(combined_metrics_raw, sort_keys=True))
    print("CALIBRATED:", json.dumps(combined_metrics_cal, sort_keys=True))
    print(f"SAVED: {args.output}")
    print("DECISION: diagnostic only; no production promotion or EntryGate changes.")


if __name__ == "__main__":
    main()
