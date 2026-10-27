"""Validate the persisted Extra Trees candidate on its untouched final holdout.

Artifact validation only. This does not promote the model, change EntryGate,
or change production thresholds/model paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.benchmark_entry_prediction_walkforward_v1 import FEATURES, LABEL, _purged_train
from scripts.train_entry_prediction_extra_trees_candidate_v1 import CANDIDATE_VERSION
from src.prediction.entry_prediction_model_v1 import (
    PRODUCTION_OOS_MIN_AUC,
    PRODUCTION_OOS_MIN_TOP_5_LIFT,
    PRODUCTION_OOS_MIN_TOP_10_LIFT,
)

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
HOLDOUT_SESSIONS = 60


def _top_lift(y_true: pd.Series, probability: np.ndarray, fraction: float) -> float:
    n = max(1, int(len(y_true) * fraction))
    order = np.argsort(-probability)[:n]
    base = float(y_true.mean())
    return float(np.asarray(y_true)[order].mean() / base) if base > 0 else 0.0


def validate(dataset_path: Path, model_path: Path) -> Dict[str, float]:
    dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    bundle = joblib.load(model_path)
    if bundle.get("model_version") != CANDIDATE_VERSION:
        raise ValueError("Unexpected candidate model version")
    if bundle.get("algorithm") != "extra_trees":
        raise ValueError("Candidate bundle is not an Extra Trees model")
    if bundle.get("production_ready") is not False:
        raise ValueError("Candidate bundle must remain production_ready=False")
    if bundle.get("promotion_status") != "candidate_only":
        raise ValueError("Candidate bundle must remain candidate_only")
    if list(bundle.get("feature_columns", [])) != list(FEATURES):
        raise ValueError("Candidate feature schema does not match the validated benchmark")

    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    dates = np.array(sorted(normalized.unique()))
    if len(dates) < HOLDOUT_SESSIONS + 1:
        raise ValueError("Dataset is too short for the untouched holdout")
    holdout_start = pd.Timestamp(dates[-HOLDOUT_SESSIONS])
    holdout = dataset[normalized >= holdout_start].copy()
    historical = _purged_train(dataset, holdout_start)
    if not historical.empty and holdout["trade_date"].dt.normalize().min() <= historical["trade_date"].dt.normalize().max():
        raise ValueError("Historical training data overlaps the holdout")

    probability = bundle["model"].predict_proba(holdout[FEATURES])[:, 1]
    metrics = {
        "rows": float(len(holdout)),
        "positive_rate": float(holdout[LABEL].mean()),
        "auc": float(roc_auc_score(holdout[LABEL], probability)),
        "ap": float(average_precision_score(holdout[LABEL], probability)),
        "brier": float(brier_score_loss(holdout[LABEL], probability)),
        "top_1pct_lift": _top_lift(holdout[LABEL], probability, 0.01),
        "top_3pct_lift": _top_lift(holdout[LABEL], probability, 0.03),
        "top_5pct_lift": _top_lift(holdout[LABEL], probability, 0.05),
        "top_10pct_lift": _top_lift(holdout[LABEL], probability, 0.10),
    }
    metrics["passes_production_gates"] = float(
        metrics["auc"] >= PRODUCTION_OOS_MIN_AUC
        and metrics["top_5pct_lift"] >= PRODUCTION_OOS_MIN_TOP_5_LIFT
        and metrics["top_10pct_lift"] >= PRODUCTION_OOS_MIN_TOP_10_LIFT
    )

    print("ENTRY_PREDICTION_EXTRA_TREES_CANDIDATE_VALIDATION")
    print(f"candidate_version={CANDIDATE_VERSION}")
    print(f"holdout_start={holdout_start.date()} holdout_end={dates[-1].date()}")
    print(f"holdout_sessions={HOLDOUT_SESSIONS} rows={len(holdout)}")
    print(f"auc={metrics['auc']:.6f} ap={metrics['ap']:.6f} brier={metrics['brier']:.6f}")
    print(
        "lifts="
        + " ".join(
            f"{key}={metrics[key]:.6f}"
            for key in ("top_1pct_lift", "top_3pct_lift", "top_5pct_lift", "top_10pct_lift")
        )
    )
    print(f"passes_production_gates={bool(metrics['passes_production_gates'])}")
    print("DECISION=Candidate artifact validated only; production integration remains blocked.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    validate(args.dataset, args.model)


if __name__ == "__main__":
    main()
