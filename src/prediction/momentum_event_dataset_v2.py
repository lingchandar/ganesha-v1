"""Build an early abnormal-momentum event dataset.

The existing momentum dataset labels generic +1.5 ATR outcomes. This v2 label
is narrower: a positive event must produce a >=2 ATR upside expansion within
3 sessions AND show abnormal volume versus the 20-session pre-event baseline.
All features remain point-in-time; only later candles are used for labels.
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
from src.prediction.momentum_event_dataset import FEATURE_COLUMNS, _build_feature_row, _safe_float

EVENT_HORIZON = 3
EVENT_ATR_MULTIPLE = 2.0
EVENT_VOLUME_MULTIPLE = 1.5


def _event_outcome(raw: pd.DataFrame, index: int, atr_value: float) -> dict:
    close = _safe_float(raw.iloc[index]["close"])
    atr = _safe_float(atr_value)
    result = {
        "event_outcome_valid": 0,
        "abnormal_up_move_3d": 0,
        "event_volume_confirmed_3d": 0,
        "early_momentum_event_3d": 0,
        "event_mfe_atr_3d": 0.0,
        "event_days_to_peak_3d": 0,
    }
    if close <= 0 or atr <= 0 or index + EVENT_HORIZON >= len(raw):
        return result

    past_volume = pd.to_numeric(raw["volume"].iloc[max(0, index - 20):index], errors="coerce")
    baseline_volume = float(past_volume.mean()) if not past_volume.empty else 0.0
    future = raw.iloc[index + 1:index + 1 + EVENT_HORIZON]
    future_high = pd.to_numeric(future["high"], errors="coerce")
    future_volume = pd.to_numeric(future["volume"], errors="coerce")
    peak = float(future_high.max())
    mfe_atr = (peak - close) / atr
    abnormal = mfe_atr >= EVENT_ATR_MULTIPLE
    volume_confirmed = baseline_volume > 0 and float(future_volume.max()) >= baseline_volume * EVENT_VOLUME_MULTIPLE
    peak_pos = int(future_high.to_numpy().argmax()) + 1 if not future_high.empty else 0

    result.update({
        "event_outcome_valid": 1,
        "abnormal_up_move_3d": int(abnormal),
        "event_volume_confirmed_3d": int(volume_confirmed),
        "early_momentum_event_3d": int(abnormal and volume_confirmed),
        "event_mfe_atr_3d": mfe_atr,
        "event_days_to_peak_3d": peak_pos,
    })
    return result


def build_dataset(output_path: Path | str = "data/prediction/momentum_event_dataset_v2.csv",
                  start_date: Optional[date] = None, end_date: Optional[date] = None) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)
    rows: list[dict] = []

    for ticker, raw_original in loader.stock_candles.items():
        if len(raw_original) < 220 + EVENT_HORIZON:
            continue
        raw = raw_original.copy()
        features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features.index = raw.index
        sector = loader.ticker_sectors.get(ticker)
        sector_series = loader.sector_series.get(sector)

        for index in range(200, len(raw) - EVENT_HORIZON):
            as_of_date = pd.Timestamp(raw.index[index]).date()
            nifty_slice, candidate_data = loader.get_point_in_time_slice(as_of_date, min_bars_required=30)
            if ticker not in candidate_data:
                continue
            row = _build_feature_row(raw, features, index, nifty_slice, sector_series)
            row.update({
                "ticker_symbol": ticker,
                "trade_date": as_of_date,
                "sector": sector or "UNKNOWN",
                **_event_outcome(raw, index, _safe_float(features.iloc[index].get("atr_14"))),
            })
            rows.append(row)

    if not rows:
        raise RuntimeError("Momentum event v2 dataset produced zero rows")
    frame = pd.DataFrame(rows).sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    required = FEATURE_COLUMNS + [
        "event_outcome_valid", "abnormal_up_move_3d", "event_volume_confirmed_3d",
        "early_momentum_event_3d", "event_mfe_atr_3d", "event_days_to_peak_3d",
    ]
    if frame[required].isna().any().any():
        raise ValueError("Momentum event v2 dataset contains NaN in required columns")
    if frame.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Momentum event v2 dataset contains duplicate ticker/date rows")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    positives = int(frame["early_momentum_event_3d"].sum())
    print(
        "momentum_event_dataset_v2="
        f"rows={len(frame)} tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()} "
        f"events={positives} event_rate={positives / len(frame):.4f} "
        f"abnormal_rate={frame['abnormal_up_move_3d'].mean():.4f} "
        f"volume_confirmed_rate={frame['event_volume_confirmed_3d'].mean():.4f}"
    )
    print(f"dataset_output={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/prediction/momentum_event_dataset_v2.csv"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    build_dataset(args.output, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
