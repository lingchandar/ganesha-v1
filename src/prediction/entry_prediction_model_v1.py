"""Walk-forward, calibrated entry prediction model v1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.engine import PredictionFeatureBuilder

MODEL_VERSION = "entry_prediction_v1"
LABEL_COLUMN = "label_target_before_stop"
PURGE_SESSIONS = 15
VALIDATION_SESSIONS = 60
MIN_TRAIN_SESSIONS = 180
TOP_FRACTIONS = (0.01, 0.03, 0.05, 0.10)


@dataclass(frozen=True)
class FoldMetrics:
    auc: float
    average_precision: float
    brier: float
    top_lift: Dict[str, float]


def _date_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["trade_date"]).dt.normalize()


def _folds(df: pd.DataFrame) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    dates = np.array(sorted(_date_series(df).unique()))
    folds = []
    cursor = MIN_TRAIN_SESSIONS
    while cursor < len(dates):
        valid_start = cursor + PURGE_SESSIONS
        valid_end = min(valid_start + VALIDATION_SESSIONS, len(dates))
        if valid_start >= len(dates):
            break
        train_dates = dates[:cursor]
        valid_dates = dates[valid_start:valid_end]
        train = df[_date_series(df).isin(train_dates)].copy()
        valid = df[_date_series(df).isin(valid_dates)].copy()
        if not train.empty and not valid.empty and train[LABEL_COLUMN].nunique() == 2 and valid[LABEL_COLUMN].nunique() == 2:
            folds.append((train, valid))
        cursor = valid_end
    if len(folds) < 3:
        raise ValueError(f"Need at least 3 chronological folds; built {len(folds)}")
    return folds


def _make_model():
    base = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=3.0,
        random_state=42,
    )
    return CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )


def _top_lift(y_true: pd.Series, probability: np.ndarray, fraction: float) -> float:
    n = max(1, int(len(y_true) * fraction))
    order = np.argsort(-probability)[:n]
    top_rate = float(np.asarray(y_true)[order].mean())
    base_rate = float(y_true.mean())
    return top_rate / base_rate if base_rate > 0 else 0.0


def evaluate_dataset(df: pd.DataFrame) -> List[FoldMetrics]:
    features = PredictionFeatureBuilder.FEATURE_COLUMNS
    folds = _folds(df)
    metrics: List[FoldMetrics] = []
    for train, valid in folds:
        model = _make_model()
        model.fit(train[features], train[LABEL_COLUMN])
        probability = model.predict_proba(valid[features])[:, 1]
        lifts = {f"top_{int(f * 100)}pct": _top_lift(valid[LABEL_COLUMN], probability, f) for f in TOP_FRACTIONS}
        metrics.append(
            FoldMetrics(
                auc=float(roc_auc_score(valid[LABEL_COLUMN], probability)),
                average_precision=float(average_precision_score(valid[LABEL_COLUMN], probability)),
                brier=float(brier_score_loss(valid[LABEL_COLUMN], probability)),
                top_lift=lifts,
            )
        )
    return metrics


def train(dataset_path: Path, model_path: Path) -> dict:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    features = PredictionFeatureBuilder.FEATURE_COLUMNS
    required = features + [LABEL_COLUMN, "ticker_symbol", "trade_date", "outcome_valid"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df[df["outcome_valid"] == 1].copy()
    df = df.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
    if len(df) < 1000:
        raise ValueError(f"Too few valid entry examples: {len(df)}")

    metrics = evaluate_dataset(df)
    mean_auc = float(np.mean([m.auc for m in metrics]))
    mean_ap = float(np.mean([m.average_precision for m in metrics]))
    mean_brier = float(np.mean([m.brier for m in metrics]))
    mean_lifts = {key: float(np.mean([m.top_lift[key] for m in metrics])) for key in metrics[0].top_lift}

    # Production is gated by out-of-sample discrimination and ranking lift.
    production_ready = mean_auc >= 0.60 and mean_lifts["top_5pct"] >= 1.50 and mean_lifts["top_10pct"] >= 1.30

    final_model = _make_model()
    final_model.fit(df[features], df[LABEL_COLUMN])
    bundle = {
        "model_version": MODEL_VERSION,
        "feature_columns": features,
        "model": final_model,
        "production_ready": production_ready,
        "mean_auc": mean_auc,
        "mean_average_precision": mean_ap,
        "mean_brier": mean_brier,
        "mean_top_lift": mean_lifts,
        "purge_sessions": PURGE_SESSIONS,
        "validation_sessions": VALIDATION_SESSIONS,
        "min_train_sessions": MIN_TRAIN_SESSIONS,
        "train_start": str(df["trade_date"].min().date()),
        "train_end": str(df["trade_date"].max().date()),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    print(f"model_version={MODEL_VERSION}")
    print(f"rows={len(df)} folds={len(metrics)}")
    print(f"mean_auc={mean_auc:.4f} mean_ap={mean_ap:.4f} mean_brier={mean_brier:.4f}")
    print("mean_top_lift=" + " ".join(f"{k}={v:.2f}" for k, v in mean_lifts.items()))
    print(f"production_ready={production_ready}")
    print(f"saved_model={model_path}")
    return bundle
