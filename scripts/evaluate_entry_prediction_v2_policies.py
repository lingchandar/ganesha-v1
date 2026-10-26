"""Walk-forward comparison of timeout label policies for Entry Prediction v2.

This script deliberately evaluates candidate label policies with the same model
family and chronological purge used by v1. It does not modify the live engine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.engine import PredictionFeatureBuilder

PURGE_SESSIONS = 15
VALIDATION_SESSIONS = 60
MIN_TRAIN_SESSIONS = 180
TOP_FRACTIONS = (0.01, 0.03, 0.05, 0.10)
POLICIES = {"0R": 0.0, "0.25R": 0.25, "0.50R": 0.50}


def folds(df: pd.DataFrame):
    dates = np.array(sorted(pd.to_datetime(df["trade_date"]).dt.normalize().unique()))
    cursor = MIN_TRAIN_SESSIONS
    while cursor < len(dates):
        valid_start = cursor + PURGE_SESSIONS
        valid_end = min(valid_start + VALIDATION_SESSIONS, len(dates))
        if valid_start >= len(dates):
            break
        train_dates = dates[:cursor]
        valid_dates = dates[valid_start:valid_end]
        train = df[pd.to_datetime(df["trade_date"]).dt.normalize().isin(train_dates)].copy()
        valid = df[pd.to_datetime(df["trade_date"]).dt.normalize().isin(valid_dates)].copy()
        if not train.empty and not valid.empty and train["policy_label"].nunique() == 2 and valid["policy_label"].nunique() == 2:
            yield train, valid
        cursor = valid_end


def model():
    base = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=3.0,
        random_state=42,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))


def top_lift(y, probability, fraction):
    n = max(1, int(len(y) * fraction))
    order = np.argsort(-probability)[:n]
    base = float(np.mean(y))
    return float(np.mean(np.asarray(y)[order]) / base) if base > 0 else 0.0


def apply_policy(df, threshold):
    out = df.copy()
    out["policy_label"] = out["label_target_before_stop"].astype(float)
    timeout = out["outcome_valid"].eq(0)
    out.loc[timeout, "policy_label"] = (out.loc[timeout, "horizon_end_r"] >= threshold).astype(float)
    return out


def evaluate(df):
    features = PredictionFeatureBuilder.FEATURE_COLUMNS
    metrics = []
    for fold_no, (train, valid) in enumerate(folds(df), start=1):
        m = model()
        m.fit(train[features], train["policy_label"])
        p = m.predict_proba(valid[features])[:, 1]
        lifts = {f"top_{int(f * 100)}pct": top_lift(valid["policy_label"], p, f) for f in TOP_FRACTIONS}
        metrics.append({
            "fold": fold_no,
            "auc": float(roc_auc_score(valid["policy_label"], p)),
            "ap": float(average_precision_score(valid["policy_label"], p)),
            "brier": float(brier_score_loss(valid["policy_label"], p)),
            **lifts,
        })
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/prediction/entry_prediction_dataset_v1.csv"))
    args = parser.parse_args()
    df = pd.read_csv(args.dataset, parse_dates=["trade_date"])
    required = PredictionFeatureBuilder.FEATURE_COLUMNS + [
        "trade_date", "ticker_symbol", "setup_type", "outcome_valid",
        "label_target_before_stop", "horizon_end_r",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.loc[df["outcome_valid"].eq(0), "horizon_end_r"].isna().any():
        raise ValueError("Unresolved rows must have horizon_end_r")

    print("ENTRY_PREDICTION_V2_POLICY_EVALUATION")
    print(f"rows={len(df)}")
    print("policy rows positive_rate folds mean_auc mean_ap mean_brier top5_lift top10_lift production_gate")

    for name, threshold in POLICIES.items():
        labeled = apply_policy(df, threshold).sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
        result = evaluate(labeled)
        mean_auc = result["auc"].mean()
        mean_ap = result["ap"].mean()
        mean_brier = result["brier"].mean()
        top5 = result["top_5pct"].mean()
        top10 = result["top_10pct"].mean()
        gate = bool(mean_auc >= 0.60 and top5 >= 1.50 and top10 >= 1.30)
        print(f"{name} {len(labeled)} {labeled['policy_label'].mean():.4f} {len(result)} {mean_auc:.4f} {mean_ap:.4f} {mean_brier:.4f} {top5:.2f} {top10:.2f} {gate}")
        print("  folds=" + "; ".join(f"{int(r.fold)}:{r.auc:.3f}/{r.top_5pct:.2f}/{r.top_10pct:.2f}" for r in result.itertuples()))

    print("\nDECISION")
    print("A policy is eligible for the next strategy-level test only if its model gate is True.")
    print("If multiple policies pass, prefer the strongest out-of-sample top-5/top-10 ranking with stable fold results.")
    print("If none pass, do not promote Entry Prediction v2 and do not touch the live engine.")


if __name__ == "__main__":
    main()
