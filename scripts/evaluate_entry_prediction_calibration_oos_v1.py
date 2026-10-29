"""Compare raw, Platt, and isotonic probability calibration on untouched OOS data.

Diagnostic only. Calibration is trained strictly on the earlier chronological
partition and evaluated only on the later partition. This script never changes
EntryGate, production thresholds, model metadata, or live order behaviour.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_LEDGER = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
RANDOM_SEED = 42


def _logit(values: pd.Series) -> np.ndarray:
    p = values.astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = pd.cut(pd.Series(np.clip(p, 0.0, 1.0)), bins=edges, include_lowest=True, labels=False)
    total = len(y)
    value = 0.0
    for idx in range(bins):
        mask = bucket.to_numpy() == idx
        if not mask.any():
            continue
        value += (mask.sum() / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(y, p)),
        "ece": _ece(y, p),
        "mean_probability": float(p.mean()),
    }
    if np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["ap"] = float(average_precision_score(y, p))
    else:
        result["auc"] = float("nan")
        result["ap"] = float("nan")
    baseline = float(y.mean())
    for fraction in (0.01, 0.03, 0.05, 0.10):
        key = f"top_{int(fraction * 100)}pct_lift"
        n = max(1, int(np.ceil(len(y) * fraction)))
        result[key] = float(np.mean(y[np.argsort(-p)[:n]]) / baseline) if baseline > 0 else float("nan")
    return result


def _fit_calibrators(train: pd.DataFrame) -> tuple[LogisticRegression, IsotonicRegression]:
    x = train["probability_target_before_stop"]
    y = train["label_target_before_stop"].astype(int)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(_logit(x), y)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(x.to_numpy(), y.to_numpy())
    return platt, isotonic


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--train-fraction", type=float, default=2.0 / 3.0)
    args = parser.parse_args(argv)
    if not args.ledger.exists():
        raise FileNotFoundError(f"Ledger not found: {args.ledger}")
    if not 0.5 <= args.train_fraction < 0.9:
        raise ValueError("--train-fraction must be >= 0.5 and < 0.9")

    frame = pd.read_csv(args.ledger)
    required = {"ticker_symbol", "trade_date", "probability_target_before_stop", "label_target_before_stop"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ledger missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)

    split = int(len(frame) * args.train_fraction)
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    if train["label_target_before_stop"].nunique() < 2 or test["label_target_before_stop"].nunique() < 2:
        raise RuntimeError("Chronological train/test split must contain both classes")

    platt, isotonic = _fit_calibrators(train)
    y_test = test["label_target_before_stop"].astype(int).to_numpy()
    raw = test["probability_target_before_stop"].astype(float).to_numpy()
    platt_p = platt.predict_proba(_logit(test["probability_target_before_stop"]))[:, 1]
    isotonic_p = isotonic.predict(test["probability_target_before_stop"].to_numpy())

    print("=== GANESHA V2 CALIBRATION + CHRONOLOGICAL OOS ===")
    print(f"ledger={args.ledger}")
    print(f"rows={len(frame)} | train={len(train)} | untouched_oos={len(test)}")
    print(f"train_date={train['trade_date'].min().date()} -> {train['trade_date'].max().date()}")
    print(f"oos_date={test['trade_date'].min().date()} -> {test['trade_date'].max().date()}")
    print(f"train_positive_rate={train['label_target_before_stop'].mean():.6f}")
    print(f"oos_positive_rate={y_test.mean():.6f}")
    print(f"platt_intercept={platt.intercept_[0]:.6f}")
    print(f"platt_slope={platt.coef_[0][0]:.6f}")

    results = {
        "RAW": _metrics(y_test, raw),
        "PLATT": _metrics(y_test, platt_p),
        "ISOTONIC": _metrics(y_test, isotonic_p),
    }
    print("\n=== UNTOUCHED OOS COMPARISON ===")
    keys = ("auc", "ap", "brier", "ece", "mean_probability", "top_1pct_lift", "top_3pct_lift", "top_5pct_lift", "top_10pct_lift")
    for name, metrics in results.items():
        print(name)
        for key in keys:
            print(f"  {key}={metrics[key]:.6f}")

    best_brier = min(results, key=lambda name: results[name]["brier"])
    best_ece = min(results, key=lambda name: results[name]["ece"])
    best_auc = max(results, key=lambda name: results[name]["auc"])
    best_top10 = max(results, key=lambda name: results[name]["top_10pct_lift"])
    print("\n=== OOS INTERPRETATION ===")
    print(f"best_brier={best_brier}")
    print(f"best_ece={best_ece}")
    print(f"best_auc={best_auc}")
    print(f"best_top10_lift={best_top10}")
    print("DECISION: diagnostic only; no calibration model is promoted automatically.")
    print("REQUIREMENT: a calibration candidate must improve probability quality on untouched OOS without materially damaging ranking/lift before shadow adoption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
