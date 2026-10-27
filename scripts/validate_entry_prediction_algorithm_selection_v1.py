"""Nested temporal validation for entry-prediction algorithm selection.

Selects an algorithm using the first N-1 walk-forward folds and evaluates the
selected algorithm exactly once on the final untouched fold. Diagnostic only:
no production model, threshold, or EntryGate is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.benchmark_entry_prediction_walkforward_v1 import (
    DEFAULT_DATASET,
    _metrics,
    _models,
    _calibration_indices,
    _purged_train,
    _folds,
    FEATURES,
    LABEL,
)
from src.prediction.entry_prediction_model_v1 import MIN_TRAIN_SESSIONS

PRODUCTION_MIN_AUC = 0.60
PRODUCTION_MIN_TOP_10_LIFT = 1.20


def evaluate(path: Path) -> None:
    dataset = pd.read_csv(path, parse_dates=["trade_date"])
    missing = sorted(set(FEATURES + [LABEL, "trade_date"]) - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    dates = pd.Index(sorted(normalized.unique()))
    fold_ranges = _folds(dates.to_numpy())
    if len(fold_ranges) < 2:
        raise ValueError("Need at least two temporal folds for nested validation")

    rows = []
    for fold_no, (holdout_start, holdout_end) in enumerate(fold_ranges, 1):
        holdout = dataset[(normalized >= holdout_start) & (normalized <= holdout_end)].copy()
        historical = _purged_train(dataset, holdout_start)
        if historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
            continue
        train_idx, calibration_idx = _calibration_indices(historical)
        for name, base_model in _models().items():
            from sklearn.calibration import CalibratedClassifierCV

            model = CalibratedClassifierCV(
                base_model,
                method="sigmoid",
                cv=[(train_idx, calibration_idx)],
                ensemble=True,
            )
            model.fit(historical[FEATURES], historical[LABEL])
            probability = model.predict_proba(holdout[FEATURES])[:, 1]
            rows.append({
                "fold": fold_no,
                "model": name,
                **_metrics(holdout[LABEL], probability),
            })

    raw = pd.DataFrame(rows)
    final_fold = int(raw["fold"].max())
    development = raw[raw["fold"] < final_fold].copy()
    final = raw[raw["fold"] == final_fold].copy()

    summary = development.groupby("model").agg(
        mean_auc=("auc", "mean"),
        mean_ap=("ap", "mean"),
        mean_brier=("brier", "mean"),
    ).reset_index()
    summary = summary.sort_values(
        ["mean_auc", "mean_ap", "mean_brier"],
        ascending=[False, False, True],
    )
    selected = str(summary.iloc[0]["model"])
    baseline = "hist_gradient_boosting"
    selected_final = final[final["model"] == selected].iloc[0]
    baseline_final = final[final["model"] == baseline].iloc[0]

    passes = (
        float(selected_final["auc"]) >= PRODUCTION_MIN_AUC
        and float(selected_final["top_10_lift"]) >= PRODUCTION_MIN_TOP_10_LIFT
        and float(selected_final["auc"]) >= float(baseline_final["auc"])
        and float(selected_final["ap"]) >= float(baseline_final["ap"])
        and float(selected_final["brier"]) <= float(baseline_final["brier"])
    )

    print("ENTRY_PREDICTION_NESTED_ALGORITHM_VALIDATION")
    print(f"development_folds=1..{final_fold - 1}")
    print(f"final_untouched_fold={final_fold}")
    print("\nDEVELOPMENT_SELECTION")
    print(summary.to_string(index=False))
    print(f"\nselected_algorithm={selected}")
    print("\nFINAL_UNTOUCHED_HOLDOUT")
    print(final.to_string(index=False))
    print("\nFINAL_COMPARISON")
    print(f"selected_auc={selected_final['auc']:.6f} baseline_auc={baseline_final['auc']:.6f}")
    print(f"selected_ap={selected_final['ap']:.6f} baseline_ap={baseline_final['ap']:.6f}")
    print(f"selected_brier={selected_final['brier']:.6f} baseline_brier={baseline_final['brier']:.6f}")
    print(f"selected_top_10_lift={selected_final['top_10_lift']:.6f} baseline_top_10_lift={baseline_final['top_10_lift']:.6f}")
    print(f"CANDIDATE_PASSES_NESTED_HOLDOUT={passes}")
    print("DECISION=Diagnostic only. Even a passing nested holdout does not change EntryGate or production thresholds.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    evaluate(args.dataset)


if __name__ == "__main__":
    main()
