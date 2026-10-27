"""Evaluate one global model with regime interaction features on a fixed holdout.

Diagnostic only. The baseline uses the original v1 feature set and the candidate
uses the augmented regime-aware feature set. Both are trained on the same purged
historical rows and evaluated on the same untouched holdout rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.prediction.entry_prediction_model_v1 import MIN_TRAIN_SESSIONS, PURGE_SESSIONS, _make_model
from src.prediction.engine import PredictionFeatureBuilder

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
LABEL = "label_target_before_stop"
BASE_FEATURES = PredictionFeatureBuilder.NUMERIC_FEATURE_COLUMNS + PredictionFeatureBuilder.SETUP_FEATURE_COLUMNS
REGIME_FEATURES = PredictionFeatureBuilder.FEATURE_COLUMNS


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


def _metrics(y: pd.Series, p: np.ndarray) -> Dict[str, float]:
    return {
        "rows": float(len(y)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def evaluate(dataset_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    required = set(REGIME_FEATURES + [LABEL, "trade_date"])
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing regime-aware features: {missing}")

    _, holdout = _split(dataset)
    historical = _purged_train(dataset, pd.Timestamp(holdout["trade_date"].min()))
    if historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
        raise ValueError("Purged historical training set is too short")

    baseline = _make_model(historical)
    baseline.fit(historical[BASE_FEATURES], historical[LABEL])
    candidate = _make_model(historical)
    candidate.fit(historical[REGIME_FEATURES], historical[LABEL])

    bp = baseline.predict_proba(holdout[BASE_FEATURES])[:, 1]
    cp = candidate.predict_proba(holdout[REGIME_FEATURES])[:, 1]
    bm = _metrics(holdout[LABEL], bp)
    cm = _metrics(holdout[LABEL], cp)

    return pd.DataFrame([{
        "holdout_start": holdout["trade_date"].min().date(),
        "holdout_end": holdout["trade_date"].max().date(),
        "historical_sessions": int(historical["trade_date"].dt.normalize().nunique()),
        "holdout_rows": int(len(holdout)),
        "holdout_positive_rate": cm["positive_rate"],
        "baseline_auc": bm["auc"],
        "candidate_auc": cm["auc"],
        "auc_delta": cm["auc"] - bm["auc"],
        "baseline_ap": bm["ap"],
        "candidate_ap": cm["ap"],
        "ap_delta": cm["ap"] - bm["ap"],
        "baseline_brier": bm["brier"],
        "candidate_brier": cm["brier"],
        "brier_delta": cm["brier"] - bm["brier"],
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    report = evaluate(args.dataset)
    print("ENTRY_PREDICTION_V3_REGIME_FEATURE_HOLDOUT")
    print(report.to_string(index=False))
    row = report.iloc[0]
    print("\nDECISION")
    if row["auc_delta"] > 0 and row["ap_delta"] > 0 and row["brier_delta"] < 0:
        print("REGIME_AWARE_GLOBAL_MODEL_HAS_HOLDOUT_EDGE")
    else:
        print("NO_HOLDOUT_EDGE")
    print("Diagnostic only: no model promotion, threshold change, or EntryGate change.")


if __name__ == "__main__":
    main()
