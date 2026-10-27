"""Evaluate regime-aware entry prediction v2 with a final chronological holdout.

Candidate subgroup selection is restricted to the first half of the existing
V1 OOS predictions. Specialized models are then trained only on historical
training rows matching the selected subgroup and evaluated on the untouched
second-half holdout. This prevents selecting a subgroup and scoring it on the
same observations.

Diagnostic only: this script does not change EntryGate or live behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.diagnose_entry_prediction_oos_v1 import (
    DEFAULT_DATASET,
    DEFAULT_PREDICTIONS,
    _merge_oos,
    _momentum_regime,
    _trend_regime,
    _volatility_regime,
)
from scripts.diagnose_entry_prediction_stable_edges_v1 import discover_candidates
from src.prediction.entry_prediction_model_v1 import MIN_TRAIN_SESSIONS, _make_model
from src.prediction.engine import PredictionFeatureBuilder

FEATURES = PredictionFeatureBuilder.FEATURE_COLUMNS
LABEL = "label_target_before_stop"
MIN_SPECIALIZED_TRAIN_ROWS = 300
MIN_HOLDOUT_ROWS = 100


def _add_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["trend_regime"] = _trend_regime(frame)
    frame["momentum_regime"] = _momentum_regime(frame)
    frame["volatility_regime"] = _volatility_regime(frame)
    return frame


def _parse_group(group: str) -> Dict[str, str]:
    return dict(part.split("=", 1) for part in group.split(" | "))


def _matches(frame: pd.DataFrame, conditions: Dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).eq(value)
    return mask


def _metrics(y: pd.Series, p: np.ndarray) -> Dict[str, float]:
    return {
        "rows": float(len(y)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        "ap": float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.nunique() == 2 else float("nan"),
    }


def _chronological_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().unique()))
    if len(dates) < 2:
        raise ValueError("Need at least two OOS dates for development/holdout split")
    midpoint = len(dates) // 2
    dev_dates = set(dates[:midpoint])
    holdout_dates = set(dates[midpoint:])
    normalized = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return frame[normalized.isin(dev_dates)].copy(), frame[normalized.isin(holdout_dates)].copy()


def evaluate(dataset_path: Path, predictions_path: Path) -> pd.DataFrame:
    oos = _add_regimes(_merge_oos(dataset_path, predictions_path))
    development, holdout = _chronological_split(oos)
    candidates = discover_candidates(development)
    if candidates.empty:
        return pd.DataFrame()

    dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    dataset = _add_regimes(dataset)
    rows: List[Dict[str, object]] = []

    for candidate in candidates.to_dict("records"):
        group = str(candidate["group"])
        conditions = _parse_group(group)
        train_mask = _matches(dataset, conditions)
        holdout_mask = _matches(holdout, conditions)
        train = dataset[train_mask].copy()
        test = holdout[holdout_mask].copy()
        train_dates = pd.to_datetime(train["trade_date"]).dt.normalize().nunique()
        if (
            len(train) < MIN_SPECIALIZED_TRAIN_ROWS
            or train_dates < MIN_TRAIN_SESSIONS
            or len(test) < MIN_HOLDOUT_ROWS
        ):
            continue
        if train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
            continue

        model = _make_model(train)
        model.fit(train[FEATURES], train[LABEL])
        probability = model.predict_proba(test[FEATURES])[:, 1]
        metrics = _metrics(test[LABEL], probability)
        rows.append(
            {
                "group": group,
                "development_rows": int(len(development[_matches(development, conditions)])),
                "holdout_rows": int(len(test)),
                "specialized_train_rows": int(len(train)),
                "specialized_train_sessions": int(train_dates),
                "holdout_positive_rate": metrics["positive_rate"],
                "holdout_auc": metrics["auc"],
                "holdout_ap": metrics["ap"],
                "holdout_brier": metrics["brier"],
                "development_pooled_auc": candidate["pooled_auc"],
                "development_top_10_lift": candidate["pooled_top_10_lift"],
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["holdout_auc", "holdout_ap"], ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    report = evaluate(args.dataset, args.predictions)
    print("ENTRY_PREDICTION_V2_REGIME_AWARE_HOLDOUT")
    if report.empty:
        print("No regime-aware candidate survived development selection and holdout-size filters.")
    else:
        print(f"candidates_evaluated={len(report)}")
        print(report.head(args.top).to_string(index=False))
    print("\nDECISION")
    print("Diagnostic only: no production promotion, threshold change, or EntryGate change.")
    print("A V2 candidate must beat the development-selected baseline on the untouched holdout before further integration work.")


if __name__ == "__main__":
    main()
