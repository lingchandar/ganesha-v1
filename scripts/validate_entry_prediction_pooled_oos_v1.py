"""Pooled out-of-sample validation for entry-prediction algorithm selection.

Diagnostic only. Rebuilds the five temporal walk-forward folds, pools all
untouched holdout predictions, and evaluates the selected Extra Trees model
against the existing HGB baseline using the production gates. No production
model, threshold, or EntryGate is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.benchmark_entry_prediction_walkforward_v1 import (
    DEFAULT_DATASET,
    FEATURES,
    LABEL,
    _calibration_indices,
    _folds,
    _metrics,
    _models,
    _purged_train,
)
from src.prediction.entry_prediction_model_v1 import (
    PRODUCTION_OOS_MIN_AUC,
    PRODUCTION_OOS_MIN_TOP_5_LIFT,
    PRODUCTION_OOS_MIN_TOP_10_LIFT,
    MIN_TRAIN_SESSIONS,
)

SELECTED_MODEL = "extra_trees"
BASELINE_MODEL = "hist_gradient_boosting"


def _pooled_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    y_values = np.asarray(y, dtype=float)
    p = np.asarray(probability, dtype=float)
    base = float(y_values.mean())
    order = np.argsort(-p)
    result = {
        "rows": float(len(y_values)),
        "positive_rate": base,
        "auc": float(roc_auc_score(y_values, p)),
        "ap": float(average_precision_score(y_values, p)),
        "brier": float(brier_score_loss(y_values, p)),
    }
    for pct, key in ((0.01, "top_1pct_lift"), (0.03, "top_3pct_lift"), (0.05, "top_5pct_lift"), (0.10, "top_10pct_lift")):
        n = max(1, int(len(y_values) * pct))
        result[key] = float(y_values[order[:n]].mean() / base) if base else 0.0
    return result


def evaluate(path: Path) -> None:
    dataset = pd.read_csv(path, parse_dates=["trade_date"])
    missing = sorted(set(FEATURES + [LABEL, "trade_date"]) - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    dates = np.array(sorted(normalized.unique()))
    fold_ranges = _folds(dates)
    if len(fold_ranges) < 2:
        raise ValueError("Need at least two temporal folds for pooled OOS validation")

    model_defs = _models()
    selected_probabilities: list[np.ndarray] = []
    baseline_probabilities: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    fold_rows: list[dict[str, object]] = []

    for fold_no, (holdout_start, holdout_end) in enumerate(fold_ranges, 1):
        holdout = dataset[(normalized >= holdout_start) & (normalized <= holdout_end)].copy()
        historical = _purged_train(dataset, holdout_start)
        if historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
            continue

        train_idx, calibration_idx = _calibration_indices(historical)
        for model_name in (BASELINE_MODEL, SELECTED_MODEL):
            model = CalibratedClassifierCV(
                model_defs[model_name],
                method="sigmoid",
                cv=[(train_idx, calibration_idx)],
                ensemble=True,
            )
            model.fit(historical[FEATURES], historical[LABEL])
            probability = model.predict_proba(holdout[FEATURES])[:, 1]
            metrics = _metrics(holdout[LABEL], probability)
            fold_rows.append({"fold": fold_no, "model": model_name, "holdout_start": holdout_start.date(), "holdout_end": holdout_end.date(), **metrics})
            if model_name == SELECTED_MODEL:
                selected_probabilities.append(probability)
            else:
                baseline_probabilities.append(probability)

        pooled_labels.append(holdout[LABEL].to_numpy(dtype=float))

    if not pooled_labels:
        raise ValueError("No valid temporal folds were evaluated")

    y = np.concatenate(pooled_labels)
    selected = _pooled_metrics(pd.Series(y), np.concatenate(selected_probabilities))
    baseline = _pooled_metrics(pd.Series(y), np.concatenate(baseline_probabilities))

    selected_passes_gates = (
        selected["auc"] >= PRODUCTION_OOS_MIN_AUC
        and selected["top_5pct_lift"] >= PRODUCTION_OOS_MIN_TOP_5_LIFT
        and selected["top_10pct_lift"] >= PRODUCTION_OOS_MIN_TOP_10_LIFT
    )
    selected_beats_baseline = (
        selected["auc"] >= baseline["auc"]
        and selected["ap"] >= baseline["ap"]
        and selected["brier"] <= baseline["brier"]
    )
    production_candidate = bool(selected_passes_gates and selected_beats_baseline)

    print("ENTRY_PREDICTION_POOLED_OOS_VALIDATION")
    print(f"folds_evaluated={len(fold_ranges)}")
    print(f"selected_model={SELECTED_MODEL}")
    print(f"production_gate_auc>={PRODUCTION_OOS_MIN_AUC:.2f}")
    print(f"production_gate_top_5_lift>={PRODUCTION_OOS_MIN_TOP_5_LIFT:.2f}")
    print(f"production_gate_top_10_lift>={PRODUCTION_OOS_MIN_TOP_10_LIFT:.2f}")
    print("\nFOLD_METRICS")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print("\nPOOLED_SELECTED")
    for key, value in selected.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")
    print("\nPOOLED_BASELINE")
    for key, value in baseline.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")
    print("\nCOMPARISON")
    print(f"auc_delta={selected['auc'] - baseline['auc']:.6f}")
    print(f"ap_delta={selected['ap'] - baseline['ap']:.6f}")
    print(f"brier_delta={selected['brier'] - baseline['brier']:.6f}")
    print(f"selected_passes_production_gates={selected_passes_gates}")
    print(f"selected_beats_baseline={selected_beats_baseline}")
    print(f"PRODUCTION_CANDIDATE={production_candidate}")
    print("DECISION=Diagnostic only. Do not change EntryGate or production thresholds from this result alone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    evaluate(args.dataset)


if __name__ == "__main__":
    main()
