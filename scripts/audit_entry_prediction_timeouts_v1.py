"""Audit unresolved entry-prediction outcomes using horizon-end return.

Analysis-only. This does not change the trading model or dataset. It answers a
specific label-quality question raised by the v1 audit: when neither the stop
nor target is hit within 15 sessions, what was the trade's mark-to-market R at
the end of the horizon?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")


def audit(dataset_path: Path) -> None:
    df = pd.read_csv(dataset_path)
    required = [
        "trade_date",
        "ticker_symbol",
        "setup_type",
        "entry_price",
        "risk_points",
        "outcome_valid",
        "label_target_before_stop",
        "mfe_r",
        "mae_r",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    unresolved = df[df["outcome_valid"] == 0].copy()
    if unresolved.empty:
        print("ENTRY_PREDICTION_V1_TIMEOUT_AUDIT")
        print("unresolved_rows=0")
        return

    loader = HistoricalDataLoader()
    loader.load_data(
        start_date=df["trade_date"].min().date(),
        end_date=df["trade_date"].max().date(),
    )

    end_r_values: list[float] = []
    missing_horizons = 0

    for row in unresolved.itertuples(index=False):
        raw = loader.stock_candles.get(row.ticker_symbol)
        if raw is None or raw.empty:
            missing_horizons += 1
            end_r_values.append(float("nan"))
            continue

        dates = pd.DatetimeIndex(raw.index)
        trade_ts = pd.Timestamp(row.trade_date)
        positions = dates.searchsorted(trade_ts)
        if positions >= len(dates) or dates[positions] != trade_ts:
            missing_horizons += 1
            end_r_values.append(float("nan"))
            continue

        end_pos = positions + 15
        if end_pos >= len(raw):
            missing_horizons += 1
            end_r_values.append(float("nan"))
            continue

        entry = float(row.entry_price)
        risk = float(row.risk_points)
        if risk <= 0:
            missing_horizons += 1
            end_r_values.append(float("nan"))
            continue

        close = float(raw.iloc[end_pos]["close_price"])
        end_r_values.append((close - entry) / risk)

    unresolved["horizon_end_r"] = end_r_values
    usable = unresolved[unresolved["horizon_end_r"].notna()].copy()

    print("ENTRY_PREDICTION_V1_TIMEOUT_AUDIT")
    print(f"unresolved_rows={len(unresolved)}")
    print(f"usable_horizon_end_rows={len(usable)} missing_horizon_rows={missing_horizons}")

    if usable.empty:
        return

    print("\nHORIZON_END_R")
    print(usable["horizon_end_r"].describe().to_string())
    print(f"nonnegative_rate={(usable['horizon_end_r'] >= 0).mean():.4f}")
    print(f"positive_0_5R_rate={(usable['horizon_end_r'] > 0.5).mean():.4f}")
    print(f"positive_1R_rate={(usable['horizon_end_r'] >= 1.0).mean():.4f}")
    print(f"negative_1R_rate={(usable['horizon_end_r'] <= -1.0).mean():.4f}")

    print("\nTIMEOUT_BY_SETUP")
    by_setup = usable.groupby("setup_type").agg(
        rows=("setup_type", "size"),
        mean_end_r=("horizon_end_r", "mean"),
        median_end_r=("horizon_end_r", "median"),
        nonnegative_rate=("horizon_end_r", lambda s: (s >= 0).mean()),
        positive_1r_rate=("horizon_end_r", lambda s: (s >= 1.0).mean()),
        negative_1r_rate=("horizon_end_r", lambda s: (s <= -1.0).mean()),
    ).sort_values("rows", ascending=False)
    print(by_setup.to_string())

    print("\nINTERPRETATION_BASELINES")
    for threshold in (-1.0, -0.5, 0.0, 0.5, 1.0):
        rate = (usable["horizon_end_r"] >= threshold).mean()
        print(f"end_r_ge_{threshold:g}={rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    audit(args.dataset)
