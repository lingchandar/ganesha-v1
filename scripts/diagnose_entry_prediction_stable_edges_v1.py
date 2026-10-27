"""Find tradable, temporally stable entry-prediction OOS subgroups.

This is a diagnostic only. It does not alter the model, thresholds, EntryGate,
or live/backtest behavior. Candidate groups must have a meaningful base rate,
positive pooled ranking lift, and repeated fold-level support before they are
considered for a future regime-specific model.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.diagnose_entry_prediction_oos_v1 import (
    DEFAULT_DATASET,
    DEFAULT_PREDICTIONS,
    KEYS,
    _merge_oos,
    _momentum_regime,
    _trend_regime,
    _volatility_regime,
)

MIN_GROUP_ROWS = 200
MIN_POSITIVE_RATE = 0.05
MIN_POOLED_AUC = 0.55
MIN_POOLED_TOP_10_LIFT = 1.20
MIN_STABLE_FOLDS = 4

REGIME_COLUMNS = (
    "trend_regime",
    "momentum_regime",
    "volatility_regime",
)
BASE_GROUP_COLUMNS = ("setup_type",) + REGIME_COLUMNS


def _top_10_lift(frame: pd.DataFrame) -> float:
    if frame.empty or frame["label_target_before_stop"].mean() <= 0:
        return 0.0
    n = max(1, int(len(frame) * 0.10))
    ranked = frame.sort_values("probability", ascending=False).head(n)
    return float(ranked["label_target_before_stop"].mean() / frame["label_target_before_stop"].mean())


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y = frame["label_target_before_stop"]
    p = frame["probability"]
    return {
        "rows": float(len(frame)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        "ap": float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.nunique() == 2 else float("nan"),
        "top_10_lift": _top_10_lift(frame),
    }


def _candidate_stability(frame: pd.DataFrame) -> Tuple[int, int, float, float]:
    fold_auc_ok = 0
    fold_lift_ok = 0
    aucs: List[float] = []
    lifts: List[float] = []
    for _, fold in frame.groupby("fold", sort=True):
        if len(fold) < 30 or fold["label_target_before_stop"].nunique() < 2:
            continue
        metrics = _metrics(fold)
        auc = metrics["auc"]
        lift = metrics["top_10_lift"]
        if np.isfinite(auc):
            aucs.append(auc)
            if auc >= 0.50:
                fold_auc_ok += 1
        if np.isfinite(lift):
            lifts.append(lift)
            if lift >= 1.0:
                fold_lift_ok += 1
    return (
        fold_auc_ok,
        fold_lift_ok,
        float(np.mean(aucs)) if aucs else float("nan"),
        float(np.mean(lifts)) if lifts else float("nan"),
    )


def _group_name(columns: Sequence[str], values: Sequence[object]) -> str:
    return " | ".join(f"{column}={value}" for column, value in zip(columns, values))


def discover_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for size in (1, 2, 3):
        for columns in combinations(BASE_GROUP_COLUMNS, size):
            grouped = frame.groupby(list(columns), sort=True, dropna=False)
            for values, part in grouped:
                if not isinstance(values, tuple):
                    values = (values,)
                if len(part) < MIN_GROUP_ROWS:
                    continue
                metrics = _metrics(part)
                if metrics["positive_rate"] < MIN_POSITIVE_RATE:
                    continue
                if not np.isfinite(metrics["auc"]) or metrics["auc"] < MIN_POOLED_AUC:
                    continue
                if metrics["top_10_lift"] < MIN_POOLED_TOP_10_LIFT:
                    continue
                fold_auc_ok, fold_lift_ok, mean_fold_auc, mean_fold_lift = _candidate_stability(part)
                if fold_auc_ok < MIN_STABLE_FOLDS or fold_lift_ok < MIN_STABLE_FOLDS:
                    continue
                rows.append(
                    {
                        "group": _group_name(columns, values),
                        "dimensions": len(columns),
                        "rows": int(metrics["rows"]),
                        "positive_rate": metrics["positive_rate"],
                        "pooled_auc": metrics["auc"],
                        "pooled_ap": metrics["ap"],
                        "pooled_brier": metrics["brier"],
                        "pooled_top_10_lift": metrics["top_10_lift"],
                        "stable_auc_folds": fold_auc_ok,
                        "stable_lift_folds": fold_lift_ok,
                        "mean_fold_auc": mean_fold_auc,
                        "mean_fold_top_10_lift": mean_fold_lift,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group", "dimensions", "rows", "positive_rate", "pooled_auc", "pooled_ap",
                "pooled_brier", "pooled_top_10_lift", "stable_auc_folds", "stable_lift_folds",
                "mean_fold_auc", "mean_fold_top_10_lift",
            ]
        )

    report = pd.DataFrame(rows)
    return report.sort_values(
        ["pooled_top_10_lift", "pooled_auc", "positive_rate", "rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    merged = _merge_oos(args.dataset, args.predictions)
    merged["trend_regime"] = _trend_regime(merged)
    merged["momentum_regime"] = _momentum_regime(merged)
    merged["volatility_regime"] = _volatility_regime(merged)

    candidates = discover_candidates(merged)
    print("ENTRY_PREDICTION_V1_STABLE_EDGE_DIAGNOSTIC")
    print(f"rows={len(merged)} folds={merged['fold'].nunique()}")
    print(f"oos_dates={merged['trade_date'].min().date()}..{merged['trade_date'].max().date()}")
    print(
        "filters="
        f"min_rows:{MIN_GROUP_ROWS} "
        f"min_positive_rate:{MIN_POSITIVE_RATE:.2f} "
        f"min_auc:{MIN_POOLED_AUC:.2f} "
        f"min_top10_lift:{MIN_POOLED_TOP_10_LIFT:.2f} "
        f"min_stable_folds:{MIN_STABLE_FOLDS}"
    )
    print(f"candidates={len(candidates)}")
    if candidates.empty:
        print("No subgroup passed all stability filters.")
    else:
        print(candidates.head(args.top).to_string(index=False))

    print("\nDECISION")
    print("Diagnostic only: no model promotion, threshold change, or EntryGate change.")
    print("A subgroup is a V2 candidate only when pooled edge and fold-level stability both pass.")


if __name__ == "__main__":
    main()
