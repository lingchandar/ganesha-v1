"""Audit the focused early-momentum event dataset before ML training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {
    "ticker_symbol", "trade_date", "sector", "early_momentum_event_3d",
    "abnormal_up_move_3d", "event_volume_confirmed_3d", "event_mfe_atr_3d",
    "event_days_to_peak_3d", "volume_ratio_20d", "volume_zscore_20d",
    "return_1d", "return_3d", "return_5d", "atr_expansion_ratio",
    "relative_strength_20d", "sector_strength_20d", "delivery_pct_ratio_20d",
}

FEATURES = [
    "volume_ratio_20d", "volume_zscore_20d", "return_1d", "return_3d",
    "return_5d", "atr_expansion_ratio", "relative_strength_20d",
    "sector_strength_20d", "delivery_pct_ratio_20d",
]


def _bucket_report(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    clean = frame[[feature, "early_momentum_event_3d"]].dropna().copy()
    if clean.empty or clean[feature].nunique() < 4:
        return pd.DataFrame()
    clean["bucket"] = pd.qcut(clean[feature], q=5, duplicates="drop")
    report = clean.groupby("bucket", observed=True)["early_momentum_event_3d"].agg(["count", "mean"])
    return report.rename(columns={"mean": "event_rate"}).reset_index().assign(feature=feature)


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

    print("=== FOCUSED MOMENTUM EVENT DATASET V2 AUDIT ===")
    print(f"rows={len(frame)}")
    print(f"tickers={frame['ticker_symbol'].nunique()}")
    print(f"start={frame['trade_date'].min().date()}")
    print(f"end={frame['trade_date'].max().date()}")
    print(f"event_rate={frame['early_momentum_event_3d'].mean():.4f}")
    print(f"abnormal_up_rate={frame['abnormal_up_move_3d'].mean():.4f}")
    print(f"volume_confirmation_rate={frame['event_volume_confirmed_3d'].mean():.4f}")
    print()

    by_ticker = frame.groupby("ticker_symbol")["early_momentum_event_3d"].agg(["count", "mean"])
    print("=== EVENT RATE BY TICKER ===")
    print(by_ticker.sort_values("mean").to_string(float_format=lambda value: f"{value:.4f}"))
    print()

    print("=== EVENT RATE BY FEATURE QUINTILE ===")
    reports = [_bucket_report(frame, feature) for feature in FEATURES]
    reports = [report for report in reports if not report.empty]
    if reports:
        print(pd.concat(reports, ignore_index=True).to_string(index=False))
    else:
        print("No usable feature bucket reports")
    print()

    print("=== EVENT OUTCOME ===")
    print(f"mfe_atr_median={frame['event_mfe_atr_3d'].median():.4f}")
    print(f"mfe_atr_p75={frame['event_mfe_atr_3d'].quantile(.75):.4f}")
    print(f"mfe_atr_p90={frame['event_mfe_atr_3d'].quantile(.90):.4f}")
    print(f"days_to_peak_mean={frame['event_days_to_peak_3d'].mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/prediction/momentum_event_dataset_v2.csv"))
    args = parser.parse_args()
    audit(args.input)


if __name__ == "__main__":
    main()
