"""Repeated walk-forward benchmark for entry-prediction algorithms.

Diagnostic only. Uses the same global feature set and label policy as v1.
No production model, threshold, or EntryGate is changed.
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
MIN_HOLDOUT_SESSIONS = 60
MAX_FOLDS = 5


def _models() -> Dict[str, object]:
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.035, max_iter=300, max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=3.0, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=12, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=12, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=42)),
    }


def _purged_train(dataset: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    dates = np.array(sorted(pd.to_datetime(dataset["trade_date"]).dt.normalize().unique()))
    holdout_idx = int(np.searchsorted(dates, holdout_start.normalize(), side="left"))
    cutoff_idx = holdout_idx - PURGE_SESSIONS
    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    return dataset[normalized.isin(set(dates[:cutoff_idx]))].copy() if cutoff_idx > 0 else dataset.iloc[0:0].copy()


def _calibration_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    dates = np.array(sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().unique()))
    calibration_start = len(dates) - CALIBRATION_SESSIONS
    train_end = calibration_start - CALIBRATION_GAP
    if train_end <= 0:
        raise ValueError("Training frame is too short for calibration")
    normalized = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return np.flatnonzero(normalized.isin(dates[:train_end]).to_numpy()), np.flatnonzero(normalized.isin(dates[calibration_start:]).to_numpy())


def _metrics(y: pd.Series, p: np.ndarray) -> Dict[str, float]:
    n = max(1, int(len(y) * 0.10))
    order = np.argsort(-p)[:n]
    base = float(y.mean())
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "top_10_lift": float(np.asarray(y)[order].mean() / base) if base else 0.0}


def _folds(dates: np.ndarray) -> list[Tuple[pd.Timestamp, pd.Timestamp]]:
    first_start = MIN_TRAIN_SESSIONS + PURGE_SESSIONS
    last_start = len(dates) - MIN_HOLDOUT_SESSIONS
    if last_start < first_start:
        return []
    starts = np.linspace(first_start, last_start, MAX_FOLDS, dtype=int)
    return [(pd.Timestamp(dates[int(start)]), pd.Timestamp(dates[int(start + MIN_HOLDOUT_SESSIONS - 1)])) for start in sorted(set(starts.tolist()))]


def evaluate(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path, parse_dates=["trade_date"])
    missing = sorted(set(FEATURES + [LABEL, "trade_date"]) - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    dates = np.array(sorted(normalized.unique()))
    fold_ranges = _folds(dates)
    rows = []
    for fold_no, (holdout_start, holdout_end) in enumerate(fold_ranges, 1):
        holdout = dataset[(normalized >= holdout_start) & (normalized <= holdout_end)].copy()
        historical = _purged_train(dataset, holdout_start)
        if historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
            continue
        for name, base_model in _models().items():
            train_idx, calibration_idx = _calibration_indices(historical)
            model = CalibratedClassifierCV(base_model, method="sigmoid", cv=[(train_idx, calibration_idx)], ensemble=True)
            model.fit(historical[FEATURES], historical[LABEL])
            p = model.predict_proba(holdout[FEATURES])[:, 1]
            rows.append({"fold": fold_no, "model": name, "holdout_start": holdout_start.date(), "holdout_end": holdout_end.date(), "train_sessions": int(historical["trade_date"].dt.normalize().nunique()), "holdout_rows": int(len(holdout)), **_metrics(holdout[LABEL], p)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    raw = evaluate(args.dataset)
    summary = raw.groupby("model").agg(folds=("fold", "count"), mean_auc=("auc", "mean"), mean_ap=("ap", "mean"), mean_brier=("brier", "mean"), mean_top_10_lift=("top_10_lift", "mean")).reset_index().sort_values(["mean_auc", "mean_ap", "mean_brier"], ascending=[False, False, True])
    print("ENTRY_PREDICTION_WALKFORWARD_ALGORITHM_BENCHMARK")
    print(raw.to_string(index=False))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nDECISION")
    for _, row in summary.iterrows():
        model_rows = raw[raw["model"] == row["model"]]
        auc_stable = int((model_rows["auc"] >= 0.60).sum())
        print(f"{row['model']}: auc>=0.60 in {auc_stable}/{len(model_rows)} folds")
    print("Diagnostic only. Promote an algorithm only after stable repeated temporal edge plus unchanged production gates.")


if __name__ == "__main__":
    main()
