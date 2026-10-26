"""Leakage-safe diagnosis of entry prediction v1 feature signal."""
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
    print(f"model auc={_auc(merged['label_target_before_stop'], merged[score_col]):.4f} ap={_ap(merged['label_target_before_stop'], merged[score_col]):.4f}")

    # Use exact chronological folds, including the 15-session purge, rather than
    # reconstructing training dates from OOS fold numbers.
    print("\nSETUP_PRIOR_BY_FOLD")
    folds = _folds(df.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True))
    setup_rows = []
    for fold_number, (train, valid) in enumerate(folds, start=1):
        valid = valid.merge(pred[KEYS + ["fold"]], on=KEYS, how="inner", validate="one_to_one")
        priors = train.groupby("setup_type")["label_target_before_stop"].mean()
        score = valid["setup_type"].map(priors).fillna(float(train["label_target_before_stop"].mean()))
        setup_rows.append((fold_number, _auc(valid["label_target_before_stop"], score), _ap(valid["label_target_before_stop"], score)))
    for fold, auc, ap in setup_rows:
        print(f"fold={fold} setup_prior_auc={auc:.4f} setup_prior_ap={ap:.4f}")

    print("\nUNIVARIATE_OOS_FEATURES")
    rows = []
    for feature in PredictionFeatureBuilder.FEATURE_COLUMNS:
        aucs = [_auc(valid["label_target_before_stop"], valid[feature]) for _, valid in merged.groupby("fold", sort=True)]
        aucs = [x for x in aucs if not np.isnan(x)]
        if aucs:
            mean_auc = float(np.mean(aucs))
            rows.append((feature, mean_auc, max(mean_auc, 1.0 - mean_auc), len(aucs)))
    rows.sort(key=lambda x: x[2], reverse=True)
    for feature, mean_auc, direction_free_auc, fold_count in rows:
        print(f"feature={feature} mean_auc={mean_auc:.4f} direction_free_auc={direction_free_auc:.4f} folds={fold_count}")

    print("\nSETUP_DISTRIBUTION")
    print(merged.groupby("setup_type")["label_target_before_stop"].agg(rows="size", positive_rate="mean").sort_values("rows", ascending=False).to_string())
    print("\nCONCLUSION_GUIDANCE")
    print("If setup priors or a small number of features beat the model, simplify before rebuilding.")
    print("If nearly every feature is near-random OOS, revisit labels/setup definitions before adding model complexity.")
    print("No production decision is made by this script.")


if __name__ == "__main__":
    main()
