"""Benchmark alternative global classifiers on the same purged untouched holdout.

Diagnostic only. No production model, threshold, or EntryGate is changed.
Every candidate uses the same point-in-time dataset, chronological split, purge gap,
and session-aware calibration split so algorithm comparisons are fair.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.prediction.entry_prediction_model_v1 import CALIBRATION_GAP, CALIBRATION_SESSIONS, MIN_TRAIN_SESSIONS, PURGE_SESSIONS
from src.prediction.engine import PredictionFeatureBuilder

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
LABEL = "label_target_before_stop"
FEATURES = PredictionFeatureBuilder.NUMERIC_FEATURE_COLUMNS + PredictionFeatureBuilder.SETUP_FEATURE_COLUMNS


def _split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().unique()))
    midpoint = len(dates) // 2
    normalized = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return frame[normalized.isin(set(dates[:midpoint]))].copy(), frame[normalized.isin(set(dates[midpoint:]))].copy()


def _purged_train(dataset: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    dates = np.array(sorted(pd.to_datetime(dataset["trade_date"]).dt.normalize().unique()))
    holdout_idx = int(np.searchsorted(dates, holdout_start.normalize(), side="left"))
    cutoff_idx = holdout_idx - PURGE_SESSIONS
    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    return dataset[normalized.isin(set(dates[:cutoff_idx]))].copy() if cutoff_idx > 0 else dataset.iloc[0:0].copy()


def _calibration_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    dates = np.array(sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().unique()))
    if len(dates) <= CALIBRATION_SESSIONS + CALIBRATION_GAP:
        raise ValueError("Training frame is too short for session-aware calibration")
    calibration_start = len(dates) - CALIBRATION_SESSIONS
    train_end = calibration_start - CALIBRATION_GAP
    normalized = pd.to_datetime(frame["trade_date"]).dt.normalize()
    train_idx = np.flatnonzero(normalized.isin(dates[:train_end]).to_numpy())
    calibration_idx = np.flatnonzero(normalized.isin(dates[calibration_start:]).to_numpy())
    return train_idx, calibration_idx


def _models() -> Dict[str, object]:
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=300, max_leaf_nodes=15,
            min_samples_leaf=35, l2_regularization=3.0, random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=12, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=12, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=42),
        ),
    }


def _calibrate(model: object, frame: pd.DataFrame) -> CalibratedClassifierCV:
    train_idx, calibration_idx = _calibration_indices(frame)
    return CalibratedClassifierCV(model, method="sigmoid", cv=[(train_idx, calibration_idx)], ensemble=True)


def _metrics(y: pd.Series, probability: np.ndarray) -> Dict[str, float]:
    n = max(1, int(len(y) * 0.10))
    order = np.argsort(-probability)[:n]
    base_rate = float(y.mean())
    top_rate = float(np.asarray(y)[order].mean())
    return {
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "top_10_lift": top_rate / base_rate if base_rate > 0 else 0.0,
    }


def evaluate(dataset_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    missing = sorted(set(FEATURES + [LABEL, "trade_date"]) - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    development, holdout = _split(dataset)
    historical = _purged_train(dataset, pd.Timestamp(holdout["trade_date"].min()))
    if historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
        raise ValueError("Purged historical training set is too short")
    rows = []
    for name, base_model in _models().items():
        model = _calibrate(base_model, historical)
        model.fit(historical[FEATURES], historical[LABEL])
        probability = model.predict_proba(holdout[FEATURES])[:, 1]
        metrics = _metrics(holdout[LABEL], probability)
        rows.append({
            "model": name,
            "holdout_start": holdout["trade_date"].min().date(),
            "holdout_end": holdout["trade_date"].max().date(),
            "historical_sessions": int(historical["trade_date"].dt.normalize().nunique()),
            "holdout_rows": int(len(holdout)),
            "holdout_positive_rate": float(holdout[LABEL].mean()),
            **metrics,
        })
    return pd.DataFrame(rows).sort_values(["auc", "ap", "brier"], ascending=[False, False, True]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    report = evaluate(args.dataset)
    print("ENTRY_PREDICTION_MODEL_ALGORITHM_BENCHMARK")
    print(report.to_string(index=False))
    print("\nDECISION")
    best = report.iloc[0]
    print(f"best_by_auc={best['model']} auc={best['auc']:.4f} ap={best['ap']:.4f} brier={best['brier']:.4f} top_10_lift={best['top_10_lift']:.2f}x")
    print("Diagnostic only: do not promote an algorithm from this benchmark without repeated temporal validation.")


if __name__ == "__main__":
    main()
