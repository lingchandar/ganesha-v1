"""Audit the historical momentum-event dataset before model training.

This script is deliberately descriptive: it does not tune thresholds or fit a
model. It reports label balance, ticker stability, and simple point-in-time
feature bucket statistics so the next modeling step is based on evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "ticker_symbol",
    "trade_date",
    "hit_up_before_down_5d",
    "hit_up_before_down_10d",
    "mfe_r_5d",
    "mae_r_5d",
    "mfe_r_10d",
    "mae_r_10d",
    "delivery_available",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "return_1d",
    "return_5d",
    "atr_expansion_ratio",
    "range_compression_10d",
    "relative_strength_20d",
    "sector_strength_20d",
    "delivery_pct_ratio_20d",
}


FEATURES = [
    "volume_ratio_20d",
    "volume_zscore_20d",
    "return_1d",
    "return_5d",
    "atr_expansion_ratio",
    "range_compression_10d",
    "relative_strength_20d",
    "sector_strength_20d",
    "delivery_pct_ratio_20d",
]


def _bucket_report(frame: pd.DataFrame, feature: str, label: str) -> pd.DataFrame:
    clean = frame[[feature, label]].dropna().copy()
    if clean.empty or clean[feature].nunique() < 4:
        return pd.DataFrame()
    clean["bucket"] = pd.qcut(clean[feature], q=5, duplicates="drop")
    report = clean.groupby("bucket", observed=True)[label].agg(["count", "mean"])
    report = report.rename(columns={"mean": "positive_rate"}).reset_index()
    report["feature"] = feature
    return report[["feature", "bucket", "count", "positive_rate"]]


def audit(path: Path) -> None:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if frame.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Duplicate ticker/date rows found")

    print("=== MOMENTUM EVENT DATASET AUDIT ===")
    print(f"rows={len(frame)}")
    print(f"tickers={frame['ticker_symbol'].nunique()}")
    print(f"start={frame['trade_date'].min().date()}")
    print(f"end={frame['trade_date'].max().date()}")
    print(f"delivery_available={frame['delivery_available'].mean():.4f}")
    print()

    for label in ("hit_up_before_down_5d", "hit_up_before_down_10d"):
        print(f"{label}: positive={int(frame[label].sum())} rate={frame[label].mean():.4f}")
    print()

    by_ticker = frame.groupby("ticker_symbol")["hit_up_before_down_5d"].agg(["count", "mean"])
    print("=== 5D LABEL BY TICKER ===")
    print(by_ticker.sort_values("mean").to_string(float_format=lambda value: f"{value:.4f}"))
    print()

    print("=== 5D LABEL BY FEATURE QUINTILE ===")
    reports = []
    for feature in FEATURES:
        report = _bucket_report(frame, feature, "hit_up_before_down_5d")
        if not report.empty:
            reports.append(report)
    if reports:
        print(pd.concat(reports, ignore_index=True).to_string(index=False))
    else:
        print("No usable feature bucket reports")
    print()

    print("=== OUTCOME DISTRIBUTION ===")
    for column in ("mfe_r_5d", "mae_r_5d", "mfe_r_10d", "mae_r_10d"):
        values = frame[column]
        print(
            f"{column}: median={values.median():.4f} "
            f"p25={values.quantile(.25):.4f} p75={values.quantile(.75):.4f} "
            f"mean={values.mean():.4f}"
        )
    print()

    # Keep the conclusion descriptive. No production threshold is selected here.
    ticker_spread = by_ticker["mean"].max() - by_ticker["mean"].min()
    print("=== MODEL READINESS ===")
    print(f"ticker_5d_positive_rate_spread={ticker_spread:.4f}")
    print("next_step=build leakage-safe walk-forward event model after reviewing this audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/prediction/momentum_event_dataset_v1.csv"))
    args = parser.parse_args()
    audit(args.input)


if __name__ == "__main__":
    main()
