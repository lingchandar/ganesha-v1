"""Train a candidate Extra Trees entry-prediction bundle.

Candidate only. This script never changes EntryGate, production thresholds, or
the production model path. The candidate is trained only after the selected
algorithm has passed pooled temporal OOS validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier

from scripts.benchmark_entry_prediction_walkforward_v1 import FEATURES, LABEL, _calibration_indices, _purged_train
from src.prediction.entry_prediction_model_v1 import MIN_TRAIN_SESSIONS, MODEL_VERSION

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
DEFAULT_OUTPUT = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
CANDIDATE_VERSION = "entry_prediction_extra_trees_candidate_v1"


def train(dataset_path: Path, output_path: Path) -> None:
    dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    required = set(FEATURES + [LABEL, "trade_date"])
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    dates = sorted(normalized.unique())
    if len(dates) < MIN_TRAIN_SESSIONS + 15 + 60:
        raise ValueError("Dataset is too short for the required purge and calibration windows")

    # Keep the same final-training convention as the validated benchmark:
    # exclude the final untouched OOS window from candidate fitting.
    holdout_start = pd.Timestamp(dates[-60])
    historical = _purged_train(dataset, holdout_start)
    if historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
        raise ValueError("Insufficient historical sessions after purge")

    train_idx, calibration_idx = _calibration_indices(historical)
    base_model = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=12,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=[(train_idx, calibration_idx)],
        ensemble=True,
    )
    model.fit(historical[FEATURES], historical[LABEL])

    bundle = {
        "model_version": CANDIDATE_VERSION,
        "base_production_model_version": MODEL_VERSION,
        "algorithm": "extra_trees",
        "label_policy": "target_before_stop_within_15_sessions_unresolved_as_loss",
        "feature_columns": FEATURES,
        "model": model,
        "training_rows": int(len(historical)),
        "training_sessions": int(historical["trade_date"].dt.normalize().nunique()),
        "excluded_final_holdout_sessions": 60,
        "production_ready": False,
        "promotion_status": "candidate_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print("ENTRY_PREDICTION_EXTRA_TREES_CANDIDATE")
    print(f"output={output_path}")
    print(f"training_rows={len(historical)}")
    print(f"training_sessions={historical['trade_date'].dt.normalize().nunique()}")
    print("production_ready=False")
    print("promotion_status=candidate_only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train(args.dataset, args.output)


if __name__ == "__main__":
    main()
