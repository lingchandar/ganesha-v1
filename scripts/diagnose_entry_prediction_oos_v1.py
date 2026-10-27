"""Diagnose fold instability and ranking direction in entry prediction v1 OOS results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
DEFAULT_PREDICTIONS = Path("data/prediction/entry_prediction_v1_oos_predictions.csv")
KEYS = ["trade_date", "ticker_symbol", "setup_type"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["trade_date"])
    pred = pd.read_csv(args.predictions, parse_dates=["trade_date"])
    df = df[df["outcome_valid"] == 1].copy()
    merged = df.merge(pred[KEYS + ["probability", "fold"]], on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError(f"Prediction merge mismatch: predictions={len(pred)} merged={len(merged)}")

    print("ENTRY_PREDICTION_V1_OOS_DIAGNOSTIC")
    print(f"rows={len(merged)} folds={merged['fold'].nunique()}")
    print(f"global_auc={roc_auc_score(merged['label_target_before_stop'], merged['probability']):.4f}")
    print(f"inverse_global_auc={roc_auc_score(merged['label_target_before_stop'], 1.0 - merged['probability']):.4f}")
    print("\nBY_FOLD")
    for fold, part in merged.groupby("fold", sort=True):
        y = part["label_target_before_stop"]
        p = part["probability"]
        auc = roc_auc_score(y, p) if y.nunique() == 2 else float("nan")
        inv = roc_auc_score(y, 1.0 - p) if y.nunique() == 2 else float("nan")
        ap = average_precision_score(y, p) if y.nunique() == 2 else float("nan")
        print(
            f"fold={int(fold)} rows={len(part)} dates={part['trade_date'].min().date()}..{part['trade_date'].max().date()} "
            f"positive_rate={y.mean():.4f} auc={auc:.4f} inverse_auc={inv:.4f} ap={ap:.4f} "
            f"mean_probability={p.mean():.4f}"
        )

    print("\nBY_SETUP")
    setup = merged.groupby("setup_type").agg(
        rows=("label_target_before_stop", "size"),
        positive_rate=("label_target_before_stop", "mean"),
        mean_probability=("probability", "mean"),
    )
    print(setup.sort_values("rows", ascending=False).to_string())

    print("\nRANKING_SANITY")
    print("If inverse_auc is materially above auc, the model is learning the opposite ranking direction.")
    print("If fold AUCs change sign sharply, the main issue is temporal instability rather than model capacity.")
    print("Do not promote this model based on mean fold metrics alone.")


if __name__ == "__main__":
    main()
