"""Report why stable V2 regime candidates cannot reach holdout evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_entry_prediction_oos_v1 import DEFAULT_DATASET, DEFAULT_PREDICTIONS, _merge_oos, _momentum_regime, _trend_regime
from scripts.diagnose_entry_prediction_stable_edges_v1 import discover_candidates
from src.prediction.entry_prediction_model_v1 import MIN_TRAIN_SESSIONS, PURGE_SESSIONS


def _cutoffs(frame):
    low, high = frame["atr_pct"].quantile([1 / 3, 2 / 3])
    return float(low), float(high)


def _regimes(frame, cutoffs):
    frame = frame.copy()
    frame["trend_regime"] = _trend_regime(frame)
    frame["momentum_regime"] = _momentum_regime(frame)
    low, high = cutoffs
    frame["volatility_regime"] = np.where(frame["atr_pct"] <= low, "low", np.where(frame["atr_pct"] <= high, "normal", "high"))
    return frame


def _parse(group):
    return dict(part.split("=", 1) for part in group.split(" | "))


def _matches(frame, conditions):
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).eq(value)
    return mask


def main():
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    oos = _merge_oos(args.dataset, args.predictions)
    dates = np.array(sorted(pd.to_datetime(oos["trade_date"]).dt.normalize().unique()))
    midpoint = len(dates) // 2
    development = oos[pd.to_datetime(oos["trade_date"]).dt.normalize().isin(set(dates[:midpoint]))].copy()
    holdout = oos[pd.to_datetime(oos["trade_date"]).dt.normalize().isin(set(dates[midpoint:]))].copy()
    cutoffs = _cutoffs(development)
    development = _regimes(development, cutoffs)
    holdout = _regimes(holdout, cutoffs)
    candidates = discover_candidates(development)

    dataset = pd.read_csv(args.dataset, parse_dates=["trade_date"])
    historical_dates = np.array(sorted(dataset["trade_date"].dt.normalize().unique()))
    holdout_start = pd.Timestamp(holdout["trade_date"].min()).normalize()
    holdout_idx = int(np.searchsorted(historical_dates, holdout_start, side="left"))
    cutoff_idx = holdout_idx - PURGE_SESSIONS
    historical = dataset[dataset["trade_date"].dt.normalize().isin(set(historical_dates[:max(0, cutoff_idx)]))].copy()
    historical = _regimes(historical, cutoffs)

    print("ENTRY_PREDICTION_V2_CANDIDATE_FEASIBILITY")
    print(f"development_dates={development['trade_date'].min().date()}..{development['trade_date'].max().date()}")
    print(f"holdout_dates={holdout['trade_date'].min().date()}..{holdout['trade_date'].max().date()}")
    print(f"historical_cutoff_dates={historical['trade_date'].min().date() if len(historical) else None}..{historical['trade_date'].max().date() if len(historical) else None}")
    print(f"historical_sessions={historical['trade_date'].dt.normalize().nunique()} required={MIN_TRAIN_SESSIONS}")
    print(f"purge_sessions={PURGE_SESSIONS}")
    print("\nCANDIDATES")
    rows = []
    for candidate in candidates.to_dict("records"):
        group = str(candidate["group"])
        conditions = _parse(group)
        train = historical[_matches(historical, conditions)]
        test = holdout[_matches(holdout, conditions)]
        train_sessions = train["trade_date"].dt.normalize().nunique()
        train_classes = train["label_target_before_stop"].nunique()
        test_classes = test["label_target_before_stop"].nunique()
        if len(train) < 300:
            reason = "train_rows"
        elif train_sessions < MIN_TRAIN_SESSIONS:
            reason = "train_sessions"
        elif len(test) < 100:
            reason = "holdout_rows"
        elif train_classes < 2:
            reason = "train_one_class"
        elif test_classes < 2:
            reason = "holdout_one_class"
        else:
            reason = "accepted"
        rows.append({"group": group, "train_rows": len(train), "train_sessions": train_sessions, "train_positive_rate": train["label_target_before_stop"].mean() if len(train) else np.nan, "holdout_rows": len(test), "holdout_positive_rate": test["label_target_before_stop"].mean() if len(test) else np.nan, "train_classes": train_classes, "holdout_classes": test_classes, "reason": reason})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nDECISION")
    print("This report diagnoses feasibility only; it does not change EntryGate or production thresholds.")


if __name__ == "__main__":
    main()
