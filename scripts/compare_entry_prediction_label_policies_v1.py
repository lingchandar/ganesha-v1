"""Compare entry-prediction label policies with identical chronological folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.prediction.entry_prediction_model_v1 import _folds
from src.prediction.engine import PredictionFeatureBuilder

DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
FEATURES = PredictionFeatureBuilder.FEATURE_COLUMNS


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=3.0,
        random_state=42,
    )


def evaluate(name: str, frame: pd.DataFrame) -> None:
    frame = frame.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
    folds = _folds(frame)
    aucs = []
    aps = []
    for train, valid in folds:
        model = make_model()
        model.fit(train[FEATURES], train["policy_label"])
        p = model.predict_proba(valid[FEATURES])[:, 1]
        aucs.append(roc_auc_score(valid["policy_label"], p))
        aps.append(average_precision_score(valid["policy_label"], p))
    print(f"policy={name} rows={len(frame)} folds={len(folds)} mean_auc={np.mean(aucs):.4f} mean_ap={np.mean(aps):.4f}")
    print("  fold_auc=" + " ".join(f"{x:.4f}" for x in aucs))


def main() -> None:
    df = pd.read_csv(DATASET, parse_dates=["trade_date"])
    df = df.copy()
    df["policy_label"] = df["label_target_before_stop"].astype(int)
    valid = df[df["outcome_valid"] == 1].copy()
    evaluate("valid_only", valid)

    conservative = df.copy()
    conservative["policy_label"] = np.where(conservative["outcome_valid"].eq(1), conservative["label_target_before_stop"], 0).astype(int)
    evaluate("unresolved_as_loss", conservative)

    resolved_or_horizon = df.copy()
    resolved_or_horizon["policy_label"] = np.where(
        resolved_or_horizon["outcome_valid"].eq(1),
        resolved_or_horizon["label_target_before_stop"],
        (resolved_or_horizon["horizon_end_r"] > 0).astype(int),
    ).astype(int)
    evaluate("horizon_end_direction_for_unresolved", resolved_or_horizon)


if __name__ == "__main__":
    main()
