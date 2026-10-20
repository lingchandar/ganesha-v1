"""Build a fast, point-in-time early abnormal-momentum event dataset.

v2 predicts whether an unusual upside move starts within the next three
sessions. The feature path is vectorized per ticker and point-in-time universe
lookups are cached by date so the dataset can be regenerated quickly.
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
from src.prediction.momentum_event_dataset import FEATURE_COLUMNS, _safe_float

EVENT_HORIZON = 3
EVENT_ATR_MULTIPLE = 2.0
EVENT_VOLUME_MULTIPLE = 1.5
MIN_HISTORY = 200


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
    volume_confirmed = (
        baseline_volume > 0
        and float(future_volume.max()) >= baseline_volume * EVENT_VOLUME_MULTIPLE
    )
    peak_pos = int(future_high.to_numpy().argmax()) + 1 if not future_high.empty else 0

    result.update(
        {
            "event_outcome_valid": 1,
            "abnormal_up_move_3d": int(abnormal),
            "event_volume_confirmed_3d": int(volume_confirmed),
            "early_momentum_event_3d": int(abnormal and volume_confirmed),
            "event_mfe_atr_3d": mfe_atr,
            "event_days_to_peak_3d": peak_pos,
        }
    )
    return result


def _vectorized_features(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    sector_series: Optional[pd.Series],
) -> pd.DataFrame:
    """Build all stock/sector features in one vectorized pass."""
    close = pd.to_numeric(raw["close"], errors="coerce")
    volume = pd.to_numeric(raw["volume"], errors="coerce")
    high = pd.to_numeric(raw["high"], errors="coerce")
    low = pd.to_numeric(raw["low"], errors="coerce")
    open_ = pd.to_numeric(raw["open"], errors="coerce")
    returns1 = close.pct_change(1)
    returns3 = close.pct_change(3)
    returns5 = close.pct_change(5)
    returns20 = close.pct_change(20)

    mean5 = volume.rolling(5, min_periods=1).mean().clip(lower=1e-9)
    mean20 = volume.rolling(20, min_periods=1).mean().clip(lower=1e-9)
    std20 = volume.rolling(20, min_periods=1).std().fillna(0.0)
    atr = pd.to_numeric(features["atr_14"], errors="coerce").fillna(0.0)
    close_feature = close.clip(lower=1e-9)

    out = pd.DataFrame(index=raw.index)
    out["return_1d"] = returns1
    out["return_3d"] = returns3
    out["return_5d"] = returns5
    out["return_acceleration_1d"] = returns1 - returns1.shift(1)
    out["return_acceleration_3d"] = returns3 - returns3.shift(3)
    out["volume_ratio_5d"] = volume / mean5
    out["volume_ratio_20d"] = volume / mean20
    out["volume_zscore_20d"] = (volume - mean20) / std20.clip(lower=1e-9)
    out["range_pct"] = (high - low) / close_feature
    out["range_atr_multiple"] = (high - low) / atr.clip(lower=1e-9)
    out["atr_pct"] = atr / close_feature
    atr_sma20 = pd.to_numeric(features["atr_sma_20"], errors="coerce").fillna(atr)
    out["atr_expansion_ratio"] = atr / atr_sma20.clip(lower=1e-9)
    out["gap_pct"] = open_ / close.shift(1).clip(lower=1e-9) - 1.0
    out["rsi_14"] = pd.to_numeric(features["rsi_14"], errors="coerce").fillna(50.0)
    out["rsi_change_3d"] = out["rsi_14"] - out["rsi_14"].shift(3).fillna(50.0)
    out["macd_hist_pct"] = pd.to_numeric(features["macd_hist"], errors="coerce").fillna(0.0) / close_feature
    out["macd_hist_change_pct"] = pd.to_numeric(features["macd_hist_diff"], errors="coerce").fillna(0.0) / close_feature
    out["adx_14"] = pd.to_numeric(features["adx_14"], errors="coerce").fillna(0.0)
    out["plus_di_minus_di"] = (
        pd.to_numeric(features["plus_di"], errors="coerce").fillna(0.0)
        - pd.to_numeric(features["minus_di"], errors="coerce").fillna(0.0)
    )

    ema20 = pd.to_numeric(features["ema_20"], errors="coerce").fillna(close)
    ema50 = pd.to_numeric(features["ema_50"], errors="coerce").fillna(close)
    ema200 = pd.to_numeric(features["ema_200"], errors="coerce").fillna(close)
    out["price_vs_ema20_pct"] = close_feature / ema20.clip(lower=1e-9) - 1.0
    out["price_vs_ema50_pct"] = close_feature / ema50.clip(lower=1e-9) - 1.0
    out["price_vs_ema200_pct"] = close_feature / ema200.clip(lower=1e-9) - 1.0
    out["ema20_vs_ema50_pct"] = ema20 / ema50.clip(lower=1e-9) - 1.0
    out["ema50_vs_ema200_pct"] = ema50 / ema200.clip(lower=1e-9) - 1.0
    out["ema50_slope_20"] = pd.to_numeric(features["ema_50_slope_20"], errors="coerce").fillna(0.0)

    out["distance_to_20d_high_pct"] = close_feature / high.rolling(20, min_periods=1).max().clip(lower=1e-9) - 1.0
    out["distance_to_50d_high_pct"] = close_feature / high.rolling(50, min_periods=1).max().clip(lower=1e-9) - 1.0
    rolling_max10 = close.rolling(10, min_periods=1).max()
    rolling_min10 = close.rolling(10, min_periods=1).min().clip(lower=1e-9)
    out["range_compression_10d"] = rolling_max10 / rolling_min10 - 1.0

    # These are stock returns. Benchmark returns are applied point-in-time
    # inside build_dataset because the benchmark slice changes by date.
    out["relative_strength_5d"] = returns5
    out["relative_strength_20d"] = returns20

    if sector_series is not None and not sector_series.empty:
        sector_close = pd.to_numeric(sector_series, errors="coerce")
        sector5 = sector_close.pct_change(5).reindex(raw.index).ffill()
        sector20 = sector_close.pct_change(20).reindex(raw.index).ffill()
        out["sector_strength_5d"] = returns5 - sector5
        out["sector_strength_20d"] = returns20 - sector20
    else:
        out["sector_strength_5d"] = 0.0
        out["sector_strength_20d"] = 0.0

    if "delivery_percentage" in raw.columns:
        delivery = pd.to_numeric(raw["delivery_percentage"], errors="coerce")
        prior_mean20 = delivery.shift(1).rolling(20, min_periods=1).mean()
        out["delivery_pct"] = delivery.fillna(0.0)
        out["delivery_pct_ratio_20d"] = (
            delivery / prior_mean20.clip(lower=1e-9)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["delivery_available"] = delivery.notna().astype(int)
    else:
        out["delivery_pct"] = 0.0
        out["delivery_pct_ratio_20d"] = 1.0
        out["delivery_available"] = 0

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_dataset(
    output_path: Path | str = "data/prediction/momentum_event_dataset_v2.csv",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)
    rows: list[dict] = []

    # All tickers share the same point-in-time universe lookup for a date.
    # Cache it once per session instead of doing the lookup for every ticker.
    pit_cache: dict[date, tuple[Optional[pd.DataFrame], dict]] = {}
    benchmark_cache: dict[date, tuple[float, float]] = {}

    for ticker, raw_original in loader.stock_candles.items():
        if len(raw_original) < MIN_HISTORY + EVENT_HORIZON:
            continue

        raw = raw_original.copy()
        features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features.index = raw.index
        sector = loader.ticker_sectors.get(ticker)
        sector_series = loader.sector_series.get(sector)
        vector_features = _vectorized_features(raw, features, sector_series)

        for index in range(MIN_HISTORY, len(raw) - EVENT_HORIZON):
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
                    **_event_outcome(raw, index, _safe_float(features.iloc[index].get("atr_14"))),
                }
            )
            rows.append(row)

    if not rows:
        raise RuntimeError("Momentum event v2 dataset produced zero rows")

    frame = pd.DataFrame(rows).sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    required = FEATURE_COLUMNS + [
        "event_outcome_valid",
        "abnormal_up_move_3d",
        "event_volume_confirmed_3d",
        "early_momentum_event_3d",
        "event_mfe_atr_3d",
        "event_days_to_peak_3d",
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
