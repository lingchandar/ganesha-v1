"""Walk-forward strategy-level validation for entry prediction v1.

This evaluator intentionally does not change live/backtest trading behavior. It
answers one question first: does entry_prediction_v1 improve candidate ranking
out of sample when candidates are ranked per trading session?
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.prediction.entry_prediction_model_v1 import LABEL_POLICY, _folds, _make_model
from src.prediction.engine import PredictionFeatureBuilder

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
TOP_K = (1, 3, 5, 10)


def _dedupe_best_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (
        frame.sort_values(["ticker_symbol", "probability"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("ticker_symbol", keep="first")
        .sort_values("probability", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def _daily_top_k_rates(frame: pd.DataFrame, top_k: int) -> List[float]:
    rates: List[float] = []
    for _, day in frame.groupby("trade_date", sort=True):
        ranked = _dedupe_best_ticker(day)
        if len(ranked) < top_k:
            continue
        rates.append(float(ranked.head(top_k)["label_target_before_stop"].mean()))
    return rates


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def evaluate(dataset_path: Path, predictions_path: Path | None = None) -> Dict[str, object]:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    features = PredictionFeatureBuilder.FEATURE_COLUMNS
    required = features + ["ticker_symbol", "trade_date", "label_target_before_stop"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Use the same complete-horizon label population as training. Unresolved
    # examples are valid negatives under LABEL_POLICY, not rows to discard.
    df = df.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
    folds = _folds(df)

    oos_parts: List[pd.DataFrame] = []
    for fold_index, (train, valid) in enumerate(folds, start=1):
        model = _make_model(train)
        model.fit(train[features], train["label_target_before_stop"])
        valid = valid.copy()
        valid["probability"] = model.predict_proba(valid[features])[:, 1]
        valid["fold"] = fold_index
        oos_parts.append(valid)

    oos = pd.concat(oos_parts, ignore_index=True)
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        oos[["trade_date", "ticker_symbol", "setup_type", "label_target_before_stop", "probability", "fold"]].to_csv(predictions_path, index=False)

    auc = float(roc_auc_score(oos["label_target_before_stop"], oos["probability"]))
    average_precision = float(average_precision_score(oos["label_target_before_stop"], oos["probability"]))
    brier = float(brier_score_loss(oos["label_target_before_stop"], oos["probability"]))
    base_rate = float(oos["label_target_before_stop"].mean())

    top_k_metrics: Dict[str, Dict[str, float]] = {}
    for top_k in TOP_K:
        daily_rates = _daily_top_k_rates(oos, top_k)
        top_rate = _mean_or_zero(daily_rates)
        top_k_metrics[f"top_{top_k}"] = {
            "daily_mean_positive_rate": top_rate,
            "daily_mean_lift": top_rate / base_rate if base_rate > 0 else 0.0,
            "days_evaluated": float(len(daily_rates)),
        }

    return {
        "model_version": "entry_prediction_v1",
        "label_policy": LABEL_POLICY,
        "rows": int(len(oos)),
        "folds": int(len(folds)),
        "oos_start": str(oos["trade_date"].min().date()),
        "oos_end": str(oos["trade_date"].max().date()),
        "auc": auc,
        "average_precision": average_precision,
        "brier": brier,
        "base_positive_rate": base_rate,
        "top_k": top_k_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions-out", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate(args.dataset, args.predictions_out)
    print(f"model_version={report['model_version']}")
    print(f"label_policy={report['label_policy']}")
    print(f"rows={report['rows']} folds={report['folds']} oos_start={report['oos_start']} oos_end={report['oos_end']}")
    print(f"oos_auc={report['auc']:.4f} oos_ap={report['average_precision']:.4f} oos_brier={report['brier']:.4f} base_positive_rate={report['base_positive_rate']:.4f}")
    for key, metrics in report["top_k"].items():
        print(f"{key}=positive_rate={metrics['daily_mean_positive_rate']:.4f} lift={metrics['daily_mean_lift']:.2f}x days={int(metrics['days_evaluated'])}")
    if args.predictions_out:
        print(f"predictions_out={args.predictions_out}")
    print("production_decision=NOT_DECIDED_BY_THIS_SCRIPT")


if __name__ == "__main__":
    main()
