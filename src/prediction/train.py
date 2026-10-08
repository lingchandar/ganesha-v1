"""Train a temporally split, calibrated entry prediction model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.engine import MODEL_VERSION, PredictionFeatureBuilder


LABEL_COLUMN = "label_target_before_stop"
LABEL_HORIZON_SESSIONS = 15
VALIDATION_FRACTION = 0.25


def _temporal_split(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by complete trading dates and purge the label horizon at the boundary."""
    dates = sorted(pd.to_datetime(df["trade_date"]).dt.normalize().unique())
    if len(dates) < 40:
        raise ValueError("Not enough unique trading dates for temporal validation")

    split_idx = int(len(dates) * (1.0 - validation_fraction))
    validation_start_idx = split_idx + LABEL_HORIZON_SESSIONS
    if validation_start_idx >= len(dates):
        raise ValueError("Temporal validation split leaves no validation dates after purge")

    train_dates = set(dates[:split_idx])
    valid_dates = set(dates[validation_start_idx:])

    train_df = df[pd.to_datetime(df["trade_date"]).dt.normalize().isin(train_dates)].copy()
    valid_df = df[pd.to_datetime(df["trade_date"]).dt.normalize().isin(valid_dates)].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Temporal split produced an empty train or validation set")
    if train_df["trade_date"].max() >= valid_df["trade_date"].min():
        raise ValueError("Temporal split is invalid: train/validation dates overlap")
    return train_df, valid_df


def train(
    dataset_path: Path,
    model_path: Path,
    validation_fraction: float = VALIDATION_FRACTION,
) -> None:
    df = (
        pd.read_csv(dataset_path, parse_dates=["trade_date"])
        .sort_values(["trade_date", "ticker_symbol", "setup_type"])
        .reset_index(drop=True)
    )
    features = PredictionFeatureBuilder.FEATURE_COLUMNS
    missing = [c for c in features + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    train_df, valid_df = _temporal_split(df, validation_fraction)

    if train_df[LABEL_COLUMN].nunique() < 2 or valid_df[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Both train and validation sets need both outcome classes")

    base = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    calibration_cv = TimeSeriesSplit(n_splits=3)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv)
    model.fit(train_df[features], train_df[LABEL_COLUMN])

    probabilities = model.predict_proba(valid_df[features])[:, 1]
    auc = roc_auc_score(valid_df[LABEL_COLUMN], probabilities)
    ap = average_precision_score(valid_df[LABEL_COLUMN], probabilities)

    print(f"train_rows={len(train_df)} validation_rows={len(valid_df)}")
    print(
        "train_dates="
        f"{train_df['trade_date'].min().date()} to {train_df['trade_date'].max().date()}"
    )
    print(
        "validation_dates="
        f"{valid_df['trade_date'].min().date()} to {valid_df['trade_date'].max().date()}"
    )
    print(f"purge_sessions={LABEL_HORIZON_SESSIONS}")
    print(f"validation_positive_rate={valid_df[LABEL_COLUMN].mean():.4f}")
    print(f"validation_auc={auc:.4f} validation_average_precision={ap:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_version": MODEL_VERSION,
            "feature_columns": features,
            "model": model,
            "validation_auc": float(auc),
            "validation_average_precision": float(ap),
            "train_end": str(train_df["trade_date"].max().date()),
            "validation_start": str(valid_df["trade_date"].min().date()),
            "purge_sessions": LABEL_HORIZON_SESSIONS,
        },
        model_path,
    )
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/prediction/entry_dataset.csv")
    parser.add_argument("--model", default="models/entry_prediction_v1.joblib")
    args = parser.parse_args()
    train(Path(args.dataset), Path(args.model))
