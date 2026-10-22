"""Audit Momentum Ignition Dataset v3 without using future labels as features."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET = Path("data/prediction/momentum_ignition_dataset_v3.csv")

CURRENT_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "range_atr_multiple",
    "atr_expansion_ratio",
    "relative_strength_20d",
    "sector_strength_20d",
    "price_vs_ema20_pct",
    "price_vs_ema50_pct",
    "distance_to_20d_high_pct",
    "distance_to_50d_high_pct",
    "delivery_pct_ratio_20d",
]

LABEL_COLUMNS = {
    "ignition_1d",
    "continuation_3d",
    "next_day_up_move_atr",
    "next_day_high_atr",
    "next_day_range_atr",
    "next_day_volume_ratio",
    "next_day_close_above_event",
    "ignition_outcome_valid",
    "continuation_outcome_valid",
    "continuation_mfe_atr_3d",
}


def quintile_rates(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[feature], errors="coerce")
    valid = frame.loc[values.notna(), ["ignition_1d"]].copy()
    valid[feature] = values.loc[valid.index]
    if valid[feature].nunique() < 5:
        return pd.DataFrame()
    valid["bucket"] = pd.qcut(valid[feature], 5, duplicates="drop")
    grouped = valid.groupby("bucket", observed=True)["ignition_1d"]
    result = grouped.agg(["count", "mean"]).rename(columns={"mean": "ignition_rate"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")

    print(f"dataset={args.dataset}")
    print(f"rows={len(frame)}")
    print(f"tickers={frame['ticker_symbol'].nunique()}")
    print(f"date_range={frame['trade_date'].min().date()} to {frame['trade_date'].max().date()}")
    print(f"ignition_rate={frame['ignition_1d'].mean():.4f}")
    print(f"continuation_rate={frame['continuation_3d'].mean():.4f}")

    print("\nCURRENT_FEATURE_QUINTILES")
    for feature in CURRENT_FEATURES:
        if feature not in frame.columns:
            continue
        result = quintile_rates(frame, feature)
        if result.empty:
            continue
        rates = ", ".join(f"{idx}: {row.ignition_rate:.4f}" for idx, row in result.iterrows())
        print(f"{feature}: {rates}")

    print("\nTOP_CURRENT_FEATURE_CORRELATIONS")
    numeric = frame[CURRENT_FEATURES + ["ignition_1d"]].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(numeric_only=True)["ignition_1d"].drop("ignition_1d").abs().sort_values(ascending=False)
    for feature, value in correlations.head(10).items():
        signed = numeric[[feature, "ignition_1d"]].corr().iloc[0, 1]
        print(f"{feature}: correlation={signed:.4f}")

    print("\nBEST_SIMPLE_FILTERS")
    filters = {
        "volume_ratio_20d>=2": frame["volume_ratio_20d"] >= 2.0,
        "volume_zscore_20d>=2": frame["volume_zscore_20d"] >= 2.0,
        "return_1d>=1%": frame["return_1d"] >= 0.01,
        "return_3d>=2%": frame["return_3d"] >= 0.02,
        "return_5d>=3%": frame["return_5d"] >= 0.03,
        "atr_expansion>=1.15": frame["atr_expansion_ratio"] >= 1.15,
        "range_atr>=1.0": frame["range_atr_multiple"] >= 1.0,
        "RS20>=2%": frame["relative_strength_20d"] >= 0.02,
        "sector_strength20>=2%": frame["sector_strength_20d"] >= 0.02,
    }
    for name, mask in filters.items():
        subset = frame.loc[mask]
        if subset.empty:
            continue
        lift = subset["ignition_1d"].mean() / frame["ignition_1d"].mean()
        print(f"{name}: rows={len(subset)} rate={subset['ignition_1d'].mean():.4f} lift={lift:.2f}x")

    print("\nLEAKAGE_CHECK")
    print("future_label_columns_excluded_from_feature_audit=", sorted(LABEL_COLUMNS))
    missing = [column for column in LABEL_COLUMNS if column not in frame.columns]
    print("missing_expected_label_columns=", missing)


if __name__ == "__main__":
    main()
