"""Diagnose entry prediction v1 OOS signal by setup and market regime."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
DEFAULT_PREDICTIONS = Path("data/prediction/entry_prediction_v1_oos_predictions.csv")
KEYS = ["trade_date", "ticker_symbol", "setup_type"]
MIN_GROUP_ROWS = 100


def _merge_oos(dataset_path: Path, predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    pred = pd.read_csv(predictions_path, parse_dates=["trade_date"])
    required = KEYS + ["label_target_before_stop"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    missing = KEYS + ["probability", "fold"]
    missing = [column for column in missing if column not in pred.columns]
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")

    # Do not filter outcome_valid: unresolved rows are complete-horizon negatives
    # under the active label policy and are part of the pooled OOS population.
    merged = df.merge(pred[KEYS + ["probability", "fold"]], on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError(f"Prediction merge mismatch: predictions={len(pred)} merged={len(merged)}")
    return merged


def _trend_regime(frame: pd.DataFrame) -> pd.Series:
    bullish = (
        (frame["price_vs_ema200_pct"] > 0)
        & (frame["ema20_vs_ema50_pct"] > 0)
        & (frame["ema50_vs_ema200_pct"] > 0)
    )
    bearish = (
        (frame["price_vs_ema200_pct"] < 0)
        & (frame["ema20_vs_ema50_pct"] < 0)
        & (frame["ema50_vs_ema200_pct"] < 0)
    )
    return pd.Series(np.where(bullish, "bullish", np.where(bearish, "bearish", "mixed")), index=frame.index)


def _momentum_regime(frame: pd.DataFrame) -> pd.Series:
    strong_positive = (
        (frame["return_5d"] > 0)
        & (frame["plus_di_minus_di"] > 0)
        & (frame["rsi_14"] >= 55)
    )
    strong_negative = (
        (frame["return_5d"] < 0)
        & (frame["plus_di_minus_di"] < 0)
        & (frame["rsi_14"] <= 45)
    )
    return pd.Series(
        np.where(strong_positive, "positive", np.where(strong_negative, "negative", "neutral")),
        index=frame.index,
    )


def _volatility_regime(frame: pd.DataFrame) -> pd.Series:
    low_cut, high_cut = frame["atr_pct"].quantile([1 / 3, 2 / 3])
    return pd.Series(
        np.where(
            frame["atr_pct"] <= low_cut,
            "low",
            np.where(frame["atr_pct"] <= high_cut, "normal", "high"),
        ),
        index=frame.index,
    )


def _group_metrics(frame: pd.DataFrame) -> Dict[str, object]:
    y = frame["label_target_before_stop"]
    p = frame["probability"]
    result: Dict[str, object] = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
    }
    if len(frame) < MIN_GROUP_ROWS or y.nunique() < 2:
        result.update({"auc": None, "ap": None, "brier": None, "top_10_lift": None})
        return result

    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))
    n = max(1, int(len(frame) * 0.10))
    ranked = frame.sort_values("probability", ascending=False).head(n)
    top_rate = float(ranked["label_target_before_stop"].mean())
    base_rate = float(y.mean())
    result.update({
        "auc": auc,
        "ap": ap,
        "brier": brier,
        "top_10_lift": top_rate / base_rate if base_rate > 0 else 0.0,
    })
    return result


def _print_group_report(frame: pd.DataFrame, column: str, title: str) -> None:
    print(f"\n{title}")
    rows = []
    for value, part in frame.groupby(column, sort=True):
        metrics = _group_metrics(part)
        metrics[column] = value
        rows.append(metrics)
    report = pd.DataFrame(rows)
    columns = [column, "rows", "positive_rate", "mean_probability", "auc", "ap", "brier", "top_10_lift"]
    print(report[columns].sort_values(["auc", "rows"], ascending=[False, False], na_position="last").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    merged = _merge_oos(args.dataset, args.predictions)
    merged["trend_regime"] = _trend_regime(merged)
    merged["momentum_regime"] = _momentum_regime(merged)
    merged["volatility_regime"] = _volatility_regime(merged)

    y = merged["label_target_before_stop"]
    p = merged["probability"]
    print("ENTRY_PREDICTION_V1_OOS_REGIME_DIAGNOSTIC")
    print(f"rows={len(merged)} folds={merged['fold'].nunique()}")
    print(f"oos_dates={merged['trade_date'].min().date()}..{merged['trade_date'].max().date()}")
    print(f"global_auc={roc_auc_score(y, p):.4f}")
    print(f"global_ap={average_precision_score(y, p):.4f}")
    print(f"global_brier={brier_score_loss(y, p):.4f}")
    print(f"global_positive_rate={y.mean():.4f}")

    _print_group_report(merged, "setup_type", "BY_SETUP")
    _print_group_report(merged, "trend_regime", "BY_TREND_REGIME")
    _print_group_report(merged, "momentum_regime", "BY_MOMENTUM_REGIME")
    _print_group_report(merged, "volatility_regime", "BY_VOLATILITY_REGIME")

    merged["setup_trend"] = merged["setup_type"] + " | " + merged["trend_regime"]
    _print_group_report(merged, "setup_trend", "BY_SETUP_AND_TREND")

    print("\nDECISION")
    print("This is diagnostic only. It does not promote the model or change EntryGate.")
    print("Use the strongest stable groups to decide whether a setup/regime-specific model or threshold is justified.")
    print("Minimum group size is 100 OOS rows; smaller groups are reported without ranking metrics.")


if __name__ == "__main__":
    main()
