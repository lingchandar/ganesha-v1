"""Compare economically meaningful timeout label policies for entry prediction v2."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.prediction.entry_prediction_model_v1 import _folds

KEYS = ["trade_date", "ticker_symbol", "setup_type"]
POLICIES = {"0R": 0.0, "0.25R": 0.25, "0.50R": 0.50}


def _auc(y: pd.Series, score: pd.Series) -> float:
    frame = pd.DataFrame({"y": y, "score": score}).dropna()
    if frame["y"].nunique() < 2 or frame["score"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(frame["y"], frame["score"]))


def _ap(y: pd.Series, score: pd.Series) -> float:
    frame = pd.DataFrame({"y": y, "score": score}).dropna()
    if frame["y"].nunique() < 2:
        return float("nan")
    return float(average_precision_score(frame["y"], frame["score"]))


def _apply_policy(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    resolved = out["outcome_valid"].eq(1)
    timeout = ~resolved
    out["policy_label"] = np.nan
    out.loc[resolved, "policy_label"] = out.loc[resolved, "label_target_before_stop"].astype(float)
    out.loc[timeout, "policy_label"] = (
        out.loc[timeout, "horizon_end_r"] >= threshold
    ).astype(float)
    return out


def _setup_prior_oos(df: pd.DataFrame) -> tuple[float, float, float]:
    frame = df.sort_values(KEYS).reset_index(drop=True)
    fold_results = []
    for train, valid in _folds(frame):
        priors = train.groupby("setup_type")["policy_label"].mean()
        score = valid["setup_type"].map(priors).fillna(train["policy_label"].mean())
        fold_results.append(
            (
                _auc(valid["policy_label"], score),
                _ap(valid["policy_label"], score),
                float((valid["policy_label"] == 1).mean()),
            )
        )
    valid_results = [x for x in fold_results if not np.isnan(x[0])]
    return (
        float(np.mean([x[0] for x in valid_results])),
        float(np.mean([x[1] for x in valid_results])),
        float(np.mean([x[2] for x in valid_results])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/prediction/entry_prediction_dataset_v1.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["trade_date"])
    required = ["trade_date", "ticker_symbol", "setup_type", "outcome_valid", "label_target_before_stop", "horizon_end_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["horizon_end_r"] = pd.to_numeric(df["horizon_end_r"], errors="coerce")
    if df.loc[df["outcome_valid"].eq(0), "horizon_end_r"].isna().any():
        raise ValueError("Unresolved rows must have horizon_end_r")

    print("ENTRY_PREDICTION_TIMEOUT_POLICY_COMPARISON")
    print(f"rows={len(df)} unresolved={int(df['outcome_valid'].eq(0).sum())}")
    print()
    print("POLICY_SUMMARY")
    print("policy positive_rate timeout_positive_rate oos_setup_prior_auc oos_setup_prior_ap")

    for name, threshold in POLICIES.items():
        labeled = _apply_policy(df, threshold)
        timeout = labeled["outcome_valid"].eq(0)
        auc, ap, oos_positive = _setup_prior_oos(labeled)
        timeout_positive = float(labeled.loc[timeout, "policy_label"].mean())
        print(f"{name} {labeled['policy_label'].mean():.4f} {timeout_positive:.4f} {auc:.4f} {ap:.4f}")

    print("\nPOLICY_BY_SETUP")
    for name, threshold in POLICIES.items():
        labeled = _apply_policy(df, threshold)
        table = labeled.groupby("setup_type")["policy_label"].agg(rows="size", positive_rate="mean")
        print(f"\n{name}")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nINTERPRETATION")
    print("This is a label-policy comparison, not a production model decision.")
    print("The setup-prior score is a deliberately simple leakage-safe baseline.")
    print("A policy must still be evaluated with the actual v2 model and strategy-level walk-forward testing before adoption.")


if __name__ == "__main__":
    main()
