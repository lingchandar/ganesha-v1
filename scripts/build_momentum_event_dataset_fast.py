"""Fast builder for the momentum-event research dataset.

The original builder performs a point-in-time universe lookup inside the
stock/session loop. This version caches those lookups once per trading day and
reuses the same Nifty benchmark slice for every ticker on that day. The
point-in-time membership rule is therefore preserved while removing thousands
of repeated DataLoader operations.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.prediction.momentum_event_dataset import (
    FEATURE_COLUMNS,
    HORIZON_10,
    HORIZON_5,
    _build_feature_row,
    _triple_barrier_outcome,
)


def build(output_path: Path, start_date: date | None = None, end_date: date | None = None) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)

    # Cache the point-in-time universe and benchmark once per session instead
    # of recomputing them for every ticker/session pair.
    eligible_by_date: dict[date, set[str]] = {}
    nifty_by_date: dict[date, pd.DataFrame] = {}
    print(f"Preparing point-in-time cache for {len(loader.trading_dates)} sessions...")
    for pos, as_of_date in enumerate(loader.trading_dates, start=1):
        nifty_slice, candidate_data = loader.get_point_in_time_slice(
            as_of_date, min_bars_required=30
        )
        eligible_by_date[as_of_date] = set(candidate_data.keys())
        nifty_by_date[as_of_date] = nifty_slice
        if pos == 1 or pos % 100 == 0 or pos == len(loader.trading_dates):
            print(f"PIT cache: {pos}/{len(loader.trading_dates)} sessions")

    rows: list[dict] = []
    tickers = sorted(loader.stock_candles.items())

    for ticker_pos, (ticker, raw_original) in enumerate(tickers, start=1):
        if len(raw_original) < 220:
            continue
        raw = raw_original.copy()
        features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features.index = raw.index
        sector = loader.ticker_sectors.get(ticker)
        sector_series = loader.sector_series.get(sector)
        ticker_rows = 0

        for index in range(200, len(raw) - HORIZON_10):
            as_of_date = pd.Timestamp(raw.index[index]).date()
            if ticker not in eligible_by_date.get(as_of_date, set()):
                continue
            feature_row = _build_feature_row(
                raw,
                features,
                index,
                nifty_by_date.get(as_of_date),
                sector_series,
            )
            outcome_5 = _triple_barrier_outcome(raw, index, HORIZON_5)
            outcome_10 = _triple_barrier_outcome(raw, index, HORIZON_10)
            feature_row.update(
                {
                    "ticker_symbol": ticker,
                    "trade_date": as_of_date,
                    "sector": sector or "UNKNOWN",
                    **outcome_5,
                    "hit_up_before_down_10d": outcome_10["hit_up_before_down_10d"],
                    "mfe_r_10d": outcome_10["mfe_r_10d"],
                    "mae_r_10d": outcome_10["mae_r_10d"],
                    "days_to_up_10d": outcome_10["days_to_up_10d"],
                }
            )
            rows.append(feature_row)
            ticker_rows += 1

        print(f"Momentum dataset: {ticker_pos}/{len(tickers)} {ticker} rows={ticker_rows}")

    if not rows:
        raise RuntimeError("Momentum event dataset produced zero rows")

    frame = pd.DataFrame(rows).sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    required = FEATURE_COLUMNS + [
        "hit_up_before_down_5d",
        "hit_up_before_down_10d",
        "mfe_r_5d",
        "mae_r_5d",
        "mfe_r_10d",
        "mae_r_10d",
    ]
    if frame[required].isna().any().any():
        raise ValueError("Momentum event dataset contains NaN in required feature/outcome columns")
    if frame.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Momentum event dataset contains duplicate ticker/date rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    positive_5 = int(frame["hit_up_before_down_5d"].sum())
    positive_10 = int(frame["hit_up_before_down_10d"].sum())
    print(
        "momentum_event_dataset="
        f"rows={len(frame)} tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()} "
        f"positive_5d={positive_5} rate_5d={positive_5 / len(frame):.4f} "
        f"positive_10d={positive_10} rate_10d={positive_10 / len(frame):.4f} "
        f"delivery_available={frame['delivery_available'].mean():.4f}"
    )
    print(f"dataset_output={output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/prediction/momentum_event_dataset_v1.csv"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    build(args.output, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
