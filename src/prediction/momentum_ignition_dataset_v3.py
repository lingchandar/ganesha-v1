"""Build a point-in-time dataset for detecting momentum ignition early.

v2 asked whether a stock would make a large move at any point over the next
three sessions. v3 changes the objective: using information available at the
close of day D, label whether the *next session* begins a meaningful upside
move with both price/range and abnormal volume confirmation.

Future bars are used only for labels. Features are inherited from the existing
point-in-time v2 feature pipeline.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.prediction.momentum_event_dataset_v2 import (
    FEATURE_COLUMNS,
    MIN_HISTORY,
    _safe_float,
    _vectorized_features,
)

IGNITION_HORIZON = 1
IGNITION_ATR_MULTIPLE = 1.0
IGNITION_VOLUME_MULTIPLE = 1.5
CONTINUATION_HORIZON = 3
CONTINUATION_ATR_MULTIPLE = 1.5


def _ignition_outcome(raw: pd.DataFrame, index: int, atr_value: float) -> dict:
    """Create strictly future-only ignition and continuation labels."""
    close = _safe_float(raw.iloc[index]["close"])
    atr = _safe_float(atr_value)
    result = {
        "ignition_outcome_valid": 0,
        "next_day_up_move_atr": 0.0,
        "next_day_high_atr": 0.0,
        "next_day_range_atr": 0.0,
        "next_day_volume_ratio": 0.0,
        "next_day_close_above_event": 0,
        "ignition_1d": 0,
        "continuation_outcome_valid": 0,
        "continuation_mfe_atr_3d": 0.0,
        "continuation_3d": 0,
    }
    if close <= 0 or atr <= 0 or index + IGNITION_HORIZON >= len(raw):
        return result

    prior_volume = pd.to_numeric(
        raw["volume"].iloc[max(0, index - 20):index], errors="coerce"
    ).dropna()
    baseline_volume = float(prior_volume.mean()) if not prior_volume.empty else 0.0
    next_bar = raw.iloc[index + 1]
    next_close = _safe_float(next_bar["close"])
    next_high = _safe_float(next_bar["high"])
    next_low = _safe_float(next_bar["low"])
    next_volume = _safe_float(next_bar["volume"])

    up_move_atr = (next_close - close) / atr
    high_atr = (next_high - close) / atr
    range_atr = (next_high - next_low) / atr
    volume_ratio = next_volume / baseline_volume if baseline_volume > 0 else 0.0
    close_above = next_close > close
    ignition = (
        high_atr >= IGNITION_ATR_MULTIPLE
        and volume_ratio >= IGNITION_VOLUME_MULTIPLE
        and close_above
    )

    result.update(
        {
            "ignition_outcome_valid": 1,
            "next_day_up_move_atr": up_move_atr,
            "next_day_high_atr": high_atr,
            "next_day_range_atr": range_atr,
            "next_day_volume_ratio": volume_ratio,
            "next_day_close_above_event": int(close_above),
            "ignition_1d": int(ignition),
        }
    )

    if index + CONTINUATION_HORIZON < len(raw):
        future = raw.iloc[index + 1:index + 1 + CONTINUATION_HORIZON]
        future_high = pd.to_numeric(future["high"], errors="coerce").dropna()
        mfe_atr = ((float(future_high.max()) - close) / atr) if not future_high.empty else 0.0
        result["continuation_outcome_valid"] = 1
        result["continuation_mfe_atr_3d"] = mfe_atr
        result["continuation_3d"] = int(ignition and mfe_atr >= CONTINUATION_ATR_MULTIPLE)

    return result


def build_dataset(
    output_path: Path | str = "data/prediction/momentum_ignition_dataset_v3.csv",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)
    rows: list[dict] = []
    pit_cache: dict[date, tuple[Optional[pd.DataFrame], dict]] = {}
    benchmark_cache: dict[date, tuple[float, float]] = {}

    for ticker, raw_original in loader.stock_candles.items():
        if len(raw_original) < MIN_HISTORY + IGNITION_HORIZON:
            continue

        raw = raw_original.copy()
        features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features.index = raw.index
        sector = loader.ticker_sectors.get(ticker)
        sector_series = loader.sector_series.get(sector)
        vector_features = _vectorized_features(raw, features, sector_series)

        for index in range(MIN_HISTORY, len(raw) - IGNITION_HORIZON):
            as_of_date = pd.Timestamp(raw.index[index]).date()
            if as_of_date not in pit_cache:
                pit_cache[as_of_date] = loader.get_point_in_time_slice(
                    as_of_date, min_bars_required=30
                )
            nifty_slice, candidate_data = pit_cache[as_of_date]
            if ticker not in candidate_data:
                continue

            if as_of_date not in benchmark_cache:
                if nifty_slice is not None and not nifty_slice.empty and "close" in nifty_slice.columns:
                    nifty_close = pd.to_numeric(nifty_slice["close"], errors="coerce")
                    benchmark_cache[as_of_date] = (
                        _safe_float(nifty_close.pct_change(5).iloc[-1]),
                        _safe_float(nifty_close.pct_change(20).iloc[-1]),
                    )
                else:
                    benchmark_cache[as_of_date] = (0.0, 0.0)
            bench5, bench20 = benchmark_cache[as_of_date]

            idx = raw.index[index]
            row = vector_features.loc[idx, FEATURE_COLUMNS].to_dict()
            row["relative_strength_5d"] = _safe_float(row["return_5d"]) - bench5
            row["relative_strength_20d"] = _safe_float(vector_features.loc[idx, "relative_strength_20d"]) - bench20
            row.update(
                {
                    "ticker_symbol": ticker,
                    "trade_date": as_of_date,
                    "sector": sector or "UNKNOWN",
                    **_ignition_outcome(raw, index, _safe_float(features.iloc[index].get("atr_14"))),
                }
            )
            rows.append(row)

    if not rows:
        raise RuntimeError("Momentum ignition v3 dataset produced zero rows")

    frame = pd.DataFrame(rows).sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    required = FEATURE_COLUMNS + [
        "ignition_outcome_valid",
        "next_day_up_move_atr",
        "next_day_high_atr",
        "next_day_range_atr",
        "next_day_volume_ratio",
        "next_day_close_above_event",
        "ignition_1d",
        "continuation_outcome_valid",
        "continuation_mfe_atr_3d",
        "continuation_3d",
    ]
    if frame[required].isna().any().any():
        raise ValueError("Momentum ignition v3 dataset contains NaN in required columns")
    if frame.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Momentum ignition v3 dataset contains duplicate ticker/date rows")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    positives = int(frame["ignition_1d"].sum())
    continuation = int(frame["continuation_3d"].sum())
    print(
        "momentum_ignition_dataset_v3="
        f"rows={len(frame)} tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()} "
        f"ignitions={positives} ignition_rate={positives / len(frame):.4f} "
        f"continuations={continuation} continuation_rate={continuation / len(frame):.4f}"
    )
    print(f"dataset_output={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/prediction/momentum_ignition_dataset_v3.csv"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    build_dataset(args.output, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
