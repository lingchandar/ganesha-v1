"""Leakage-safe diagnosis of entry prediction v1 feature signal.

This analysis does not train or modify the production model. It answers whether
individual features or setup priors contain useful out-of-sample signal and
whether the current model is adding information beyond simple baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.prediction.entry_prediction_model_v1 import _folds
from src.prediction.engine import PredictionFeatureBuilder

KEYS = ["trade_date", "ticker_symbol", "setup_type"]


def _auc(y: pd.Series, score: pd.Series) -> float:
    frame = pd.DataFrame({"y": y, "score": score}).dropna()
    if frame.empty or frame["y"].nunique() < 2 or frame["score"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(frame["y"], frame["score"]))


def _ap(y: pd.Series, score: pd.Series) -> float:
    frame = pd.DataFrame({"y": y, "score": score}).dropna()
    if frame.empty or frame["y"].nunique() < 2:
        return float("nan")
    return float(average_precision_score(frame["y"], frame["score"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/prediction/entry_prediction_dataset_v1.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["trade_date"])
    pred = pd.read_csv(args.predictions, parse_dates=["trade_date"])
    score_col = "probability" if "probability" in pred.columns else "prediction"
    if score_col not in pred.columns:
        raise ValueError("Predictions file must contain 'probability' or 'prediction'")

    df = df[df["outcome_valid"] == 1].copy()
    merged = df.merge(pred[KEYS + ["fold", score_col]], on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError(f"Prediction merge mismatch: predictions={len(pred)} merged={len(merged)}")

    print("ENTRY_PREDICTION_V1_FEATURE_ANALYSIS")
    print(f"rows={len(merged)} features={len(PredictionFeatureBuilder.FEATURE_COLUMNS)}")
    print("\nMODEL_VS_SIMPLE_BASELINES")
    model_auc = _auc(merged["label_target_before_stop"], merged[score_col])
    model_ap = _ap(merged["label_target_before_stop"], merged[score_col])
    print(f"model auc={model_auc:.4f} ap={model_ap:.4f}")

    # Leakage-safe setup prior: estimate setup success rates using each fold's training set.
    print("\nSETUP_PRIOR_BY_FOLD")
    setup_rows = []
    for fold, valid in merged.groupby("fold", sort=True):
        train_dates = sorted(merged.loc[merged["fold"] < fold, "trade_date"].unique())
        train = df[df["trade_date"].isin(train_dates)] if train_dates else df.iloc[0:0]
        if train.empty:
            continue
        priors = train.groupby("setup_type")["label_target_before_stop"].mean()
        score = valid["setup_type"].map(priors).fillna(train["label_target_before_stop"].mean())
        setup_rows.append((fold, _auc(valid["label_target_before_stop"], score), _ap(valid["label_target_before_stop"], score)))
    for fold, auc, ap in setup_rows:
        print(f"fold={fold} setup_prior_auc={auc:.4f} setup_prior_ap={ap:.4f}")

    print("\nUNIVARIATE_OOS_FEATURES")
    rows = []
    for feature in PredictionFeatureBuilder.FEATURE_COLUMNS:
        aucs = []
        for fold, valid in merged.groupby("fold", sort=True):
            # Feature is point-in-time and evaluated only on that fold's OOS rows.
            aucs.append(_auc(valid["label_target_before_stop"], valid[feature]))
        aucs = [x for x in aucs if not np.isnan(x)]
        if aucs:
            strength = max(float(np.mean(aucs)), 1.0 - float(np.mean(aucs)))
            rows.append((feature, float(np.mean(aucs)), strength, len(aucs)))
    rows.sort(key=lambda x: x[2], reverse=True)
    for feature, mean_auc, direction_free_auc, folds in rows:
        print(f"feature={feature} mean_auc={mean_auc:.4f} direction_free_auc={direction_free_auc:.4f} folds={folds}")

    print("\nSETUP_DISTRIBUTION")
    print(
        merged.groupby("setup_type")["label_target_before_stop"]
        .agg(rows="size", positive_rate="mean")
        .sort_values("rows", ascending=False)
        .to_string()
    )

    print("\nCONCLUSION_GUIDANCE")
    print("If setup priors or a small number of features beat the model, simplify before rebuilding.")
    print("If nearly every feature is near-random OOS, revisit labels/setup definitions before adding model complexity.")
    print("No production decision is made by this script.")


if __name__ == "__main__":
    main()
