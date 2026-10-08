"""Train a temporally split, calibrated entry prediction model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.prediction.engine import MODEL_VERSION, PredictionFeatureBuilder


def train(dataset_path: Path, model_path: Path, validation_fraction: float = 0.25) -> None:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    features = PredictionFeatureBuilder.FEATURE_COLUMNS
    missing = [c for c in features + ["label_target_before_stop"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    split_idx = int(len(df) * (1.0 - validation_fraction))
    train_df, valid_df = df.iloc[:split_idx], df.iloc[split_idx:]
    if train_df["trade_date"].max() >= valid_df["trade_date"].min():
        raise ValueError("Temporal split is invalid")
    if train_df["label_target_before_stop"].nunique() < 2 or valid_df["label_target_before_stop"].nunique() < 2:
        raise ValueError("Both train and validation sets need both outcome classes")
    base = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42,
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(train_df[features], train_df["label_target_before_stop"])
    probabilities = model.predict_proba(valid_df[features])[:, 1]
    auc = roc_auc_score(valid_df["label_target_before_stop"], probabilities)
    ap = average_precision_score(valid_df["label_target_before_stop"], probabilities)
    print(f"train_rows={len(train_df)} validation_rows={len(valid_df)}")
    print(f"validation_auc={auc:.4f} validation_average_precision={ap:.4f}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_version": MODEL_VERSION,
        "feature_columns": features,
        "model": model,
        "validation_auc": float(auc),
        "validation_average_precision": float(ap),
        "train_end": str(train_df["trade_date"].max().date()),
        "validation_start": str(valid_df["trade_date"].min().date()),
    }, model_path)
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/prediction/entry_dataset.csv")
    parser.add_argument("--model", default="models/entry_prediction_v1.joblib")
    args = parser.parse_args()
    train(Path(args.dataset), Path(args.model))
