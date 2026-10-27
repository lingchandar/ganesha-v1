#!/usr/bin/env python3
"""Compare raw vs chronologically calibrated Trend Pullback probabilities.

This is diagnostic only. It never changes EntryGate, production thresholds,
or the candidate model artifact.
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


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if not mask.any():
            continue
        value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def metrics(frame: pd.DataFrame, prob_col: str) -> dict:
    y = frame["actual_success"].to_numpy(dtype=int)
    p = frame[prob_col].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
    result = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece(y, p),
    }
    if len(np.unique(y)) == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["ap"] = float(average_precision_score(y, p))
    else:
        result["auc"] = None
        result["ap"] = None
    for pct, name in [(0.01, "top_1pct_lift"), (0.03, "top_3pct_lift"), (0.05, "top_5pct_lift"), (0.10, "top_10pct_lift")]:
        n = max(1, int(np.ceil(len(frame) * pct)))
        top_rate = y[np.argsort(-p)[:n]].mean()
        result[name] = float(top_rate / y.mean()) if y.mean() > 0 else None
    return result


def fit_platt(train: pd.DataFrame, prob_col: str) -> LogisticRegression:
    x = train[[prob_col]].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
    y = train["actual_success"].to_numpy(dtype=int)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(np.log(x / (1.0 - x)), y)
    return model


def apply_platt(model: LogisticRegression, frame: pd.DataFrame, prob_col: str) -> np.ndarray:
    p = frame[prob_col].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--holdout-fraction", type=float, default=1 / 3)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("data/prediction/trend_pullback_calibration_oos_v1.json"))
    args = ap.parse_args()

    frame = pd.read_csv(args.ledger)
    required = {"date", "probability_target_before_stop", "actual_success"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["probability_target_before_stop"] = pd.to_numeric(frame["probability_target_before_stop"], errors="coerce")
    frame["actual_success"] = pd.to_numeric(frame["actual_success"], errors="coerce")
    frame = frame.dropna(subset=["date", "probability_target_before_stop", "actual_success"]).copy()
    frame["actual_success"] = frame["actual_success"].astype(int)
    frame = frame.sort_values("date").reset_index(drop=True)

    if not 0.1 <= args.holdout_fraction <= 0.5:
        raise ValueError("--holdout-fraction must be between 0.1 and 0.5")
    split = int(len(frame) * (1.0 - args.holdout_fraction))
    train = frame.iloc[:split].copy()
    holdout = frame.iloc[split:].copy()
    if train["actual_success"].nunique() < 2 or holdout["actual_success"].nunique() < 2:
        raise RuntimeError("Chronological split must contain both outcome classes in train and holdout")

    prob_col = "probability_target_before_stop"
    calibrator = fit_platt(train, prob_col)
    holdout["calibrated_probability"] = apply_platt(calibrator, holdout, prob_col)

    raw = metrics(holdout, prob_col)
    calibrated = metrics(holdout, "calibrated_probability")
    result = {
        "experiment": "trend_pullback_probability_calibration_oos_v1",
        "decision": "diagnostic_only",
        "ledger": str(args.ledger),
        "rows": int(len(frame)),
        "date_range": [frame["date"].min().date().isoformat(), frame["date"].max().date().isoformat()],
        "chronological_split": {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train_end": train["date"].max().date().isoformat(),
            "holdout_start": holdout["date"].min().date().isoformat(),
            "holdout_end": holdout["date"].max().date().isoformat(),
        },
        "platt_calibrator": {
            "intercept": float(calibrator.intercept_[0]),
            "slope": float(calibrator.coef_[0][0]),
        },
        "holdout_raw": raw,
        "holdout_calibrated": calibrated,
        "calibration_improvement": {
            "brier_delta_calibrated_minus_raw": calibrated["brier"] - raw["brier"],
            "ece_delta_calibrated_minus_raw": calibrated["ece"] - raw["ece"],
            "log_loss_delta_calibrated_minus_raw": calibrated["log_loss"] - raw["log_loss"],
            "auc_delta_calibrated_minus_raw": (calibrated["auc"] - raw["auc"]) if raw["auc"] is not None else None,
            "top_10pct_lift_delta_calibrated_minus_raw": calibrated["top_10pct_lift"] - raw["top_10pct_lift"],
        },
        "production_ready": False,
        "promotion_status": "candidate_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== TREND PULLBACK CALIBRATION OOS V1 ===")
    print(f"ROWS: {len(frame)} | TRAIN: {len(train)} | HOLDOUT: {len(holdout)}")
    print(f"DATE SPLIT: {train['date'].max().date()} -> {holdout['date'].min().date()}")
    print(f"PLATT: intercept={calibrator.intercept_[0]:.6f} slope={calibrator.coef_[0][0]:.6f}")
    print("RAW HOLDOUT:", json.dumps(raw, sort_keys=True))
    print("CALIBRATED HOLDOUT:", json.dumps(calibrated, sort_keys=True))
    print("DELTA:", json.dumps(result["calibration_improvement"], sort_keys=True))
    print(f"SAVED: {args.output}")
    print("DECISION: diagnostic only; no production promotion or EntryGate changes.")


if __name__ == "__main__":
    main()
