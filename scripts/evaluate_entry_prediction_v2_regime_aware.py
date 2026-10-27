"""Evaluate regime-aware entry prediction v2 with leakage-safe holdout validation.

Candidate subgroup selection uses only the development half of V1 OOS predictions.
Models are trained only on historical rows ending before a purged gap ahead of
the untouched holdout. Volatility regime cutoffs are learned from development
data and then frozen for training/holdout rows.

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
)
from scripts.diagnose_entry_prediction_stable_edges_v1 import discover_candidates
from src.prediction.entry_prediction_model_v1 import MIN_TRAIN_SESSIONS, PURGE_SESSIONS, _make_model
from src.prediction.engine import PredictionFeatureBuilder

FEATURES = PredictionFeatureBuilder.FEATURE_COLUMNS
LABEL = "label_target_before_stop"
MIN_SPECIALIZED_TRAIN_ROWS = 300
MIN_HOLDOUT_ROWS = 100


def _volatility_cutoffs(frame: pd.DataFrame) -> Tuple[float, float]:
    low, high = frame["atr_pct"].quantile([1 / 3, 2 / 3])
    return float(low), float(high)


def _add_regimes(frame: pd.DataFrame, volatility_cutoffs: Tuple[float, float] | None = None) -> pd.DataFrame:
    frame = frame.copy()
    frame["trend_regime"] = _trend_regime(frame)
    frame["momentum_regime"] = _momentum_regime(frame)
    low_cut, high_cut = volatility_cutoffs or _volatility_cutoffs(frame)
    frame["volatility_regime"] = np.where(frame["atr_pct"] <= low_cut, "low", np.where(frame["atr_pct"] <= high_cut, "normal", "high"))
    return frame


def _parse_group(group: str) -> Dict[str, str]:
    return dict(part.split("=", 1) for part in group.split(" | "))


def _matches(frame: pd.DataFrame, conditions: Dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).eq(value)
    return mask


def _metrics(y: pd.Series, p: np.ndarray) -> Dict[str, float]:
    return {"rows": float(len(y)), "positive_rate": float(y.mean()), "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"), "ap": float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan"), "brier": float(brier_score_loss(y, p)) if y.nunique() == 2 else float("nan")}


def _chronological_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().unique()))
    if len(dates) < 4:
        raise ValueError("Need at least four OOS dates for development/holdout split")
    midpoint = len(dates) // 2
    normalized = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return frame[normalized.isin(set(dates[:midpoint]))].copy(), frame[normalized.isin(set(dates[midpoint:]))].copy()


def _purged_historical_training(dataset: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    dates = np.array(sorted(pd.to_datetime(dataset["trade_date"]).dt.normalize().unique()))
    holdout_idx = int(np.searchsorted(dates, holdout_start.normalize(), side="left"))
    cutoff_idx = holdout_idx - PURGE_SESSIONS
    if cutoff_idx <= 0:
        return dataset.iloc[0:0].copy()
    normalized = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    return dataset[normalized.isin(set(dates[:cutoff_idx]))].copy()


def evaluate(dataset_path: Path, predictions_path: Path) -> pd.DataFrame:
    development, holdout = _chronological_split(_merge_oos(dataset_path, predictions_path))
    cutoffs = _volatility_cutoffs(development)
    development = _add_regimes(development, cutoffs)
    holdout = _add_regimes(holdout, cutoffs)
    candidates = discover_candidates(development)
    if candidates.empty:
        return pd.DataFrame()

    dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    historical = _add_regimes(_purged_historical_training(dataset, pd.Timestamp(holdout["trade_date"].min())), cutoffs)
    if historical.empty or historical["trade_date"].dt.normalize().nunique() < MIN_TRAIN_SESSIONS:
        return pd.DataFrame()

    baseline_model = _make_model(historical)
    baseline_model.fit(historical[FEATURES], historical[LABEL])
    rows: List[Dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        conditions = _parse_group(str(candidate["group"]))
        train = historical[_matches(historical, conditions)].copy()
        test = holdout[_matches(holdout, conditions)].copy()
        train_dates = pd.to_datetime(train["trade_date"]).dt.normalize().nunique()
        if len(train) < MIN_SPECIALIZED_TRAIN_ROWS or train_dates < MIN_TRAIN_SESSIONS or len(test) < MIN_HOLDOUT_ROWS or train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
            continue
        specialized_model = _make_model(train)
        specialized_model.fit(train[FEATURES], train[LABEL])
        sp = specialized_model.predict_proba(test[FEATURES])[:, 1]
        bp = baseline_model.predict_proba(test[FEATURES])[:, 1]
        sm, bm = _metrics(test[LABEL], sp), _metrics(test[LABEL], bp)
        rows.append({"group": str(candidate["group"]), "development_rows": int(len(development[_matches(development, conditions)])), "holdout_rows": int(len(test)), "specialized_train_rows": int(len(train)), "specialized_train_sessions": int(train_dates), "holdout_positive_rate": sm["positive_rate"], "specialized_holdout_auc": sm["auc"], "specialized_holdout_ap": sm["ap"], "specialized_holdout_brier": sm["brier"], "baseline_holdout_auc": bm["auc"], "baseline_holdout_ap": bm["ap"], "baseline_holdout_brier": bm["brier"], "auc_delta": sm["auc"] - bm["auc"], "ap_delta": sm["ap"] - bm["ap"], "development_pooled_auc": candidate["pooled_auc"], "development_top_10_lift": candidate["pooled_top_10_lift"]})
    return pd.DataFrame(rows).sort_values(["auc_delta", "ap_delta"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    report = evaluate(args.dataset, args.predictions)
    print("ENTRY_PREDICTION_V2_REGIME_AWARE_HOLDOUT")
    if report.empty:
        print("No regime-aware candidate survived leakage-safe development/holdout filters.")
    else:
        print(f"candidates_evaluated={len(report)}")
        print(report.head(args.top).to_string(index=False))
    print("\nDECISION")
    print("Diagnostic only: no production promotion, threshold change, or EntryGate change.")
    print("V2 is a candidate only if the specialized model beats the same-holdout global baseline.")


if __name__ == "__main__":
    main()
