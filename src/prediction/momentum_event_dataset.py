"""Build a point-in-time dataset for early momentum-event prediction.

This dataset is intentionally independent of the existing setup/entry dataset.
One row represents one stock on one completed session. Features use information
known at that session close only. Outcomes inspect only later sessions.
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


FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_acceleration_1d",
    "return_acceleration_3d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "range_pct",
    "range_atr_multiple",
    "atr_pct",
    "atr_expansion_ratio",
    "gap_pct",
    "rsi_14",
    "rsi_change_3d",
    "macd_hist_pct",
    "macd_hist_change_pct",
    "adx_14",
    "plus_di_minus_di",
    "price_vs_ema20_pct",
    "price_vs_ema50_pct",
    "price_vs_ema200_pct",
    "ema20_vs_ema50_pct",
    "ema50_vs_ema200_pct",
    "ema50_slope_20",
    "distance_to_20d_high_pct",
    "distance_to_50d_high_pct",
    "range_compression_10d",
    "relative_strength_5d",
    "relative_strength_20d",
    "sector_strength_5d",
    "sector_strength_20d",
    "delivery_pct",
    "delivery_pct_ratio_20d",
    "delivery_available",
]

HORIZON_5 = 5
HORIZON_10 = 10
UP_BARRIER_ATR = 1.5
DOWN_BARRIER_ATR = 1.0


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    value = series.pct_change(periods).iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _relative_strength(stock: pd.Series, benchmark: Optional[pd.Series], periods: int) -> float:
    if benchmark is None or benchmark.empty:
        return 0.0
    common = pd.concat([stock, benchmark], axis=1, join="inner").dropna()
    if len(common) <= periods:
        return 0.0
    stock_ret = common.iloc[:, 0].pct_change(periods).iloc[-1]
    benchmark_ret = common.iloc[:, 1].pct_change(periods).iloc[-1]
    if pd.isna(stock_ret) or pd.isna(benchmark_ret):
        return 0.0
    return float(stock_ret - benchmark_ret)


def _triple_barrier_outcome(
    raw: pd.DataFrame,
    index: int,
    horizon: int,
    up_atr: float = UP_BARRIER_ATR,
    down_atr: float = DOWN_BARRIER_ATR,
) -> dict:
    """Measure future opportunity using ATR-normalized barriers.

    The entry reference is the close at the feature timestamp. The first later
    session is inspected, so the feature row never contains future information.
    If both barriers are touched in the same daily candle, the adverse barrier
    is treated as first (conservative daily-bar convention).
    """
    close = _safe_float(raw.iloc[index]["close"])
    atr = _safe_float(raw.iloc[index].get("atr_14"), 0.0)
    if close <= 0 or atr <= 0:
        return {
            "outcome_valid": 0,
            "hit_up_before_down_5d": 0,
            "hit_up_before_down_10d": 0,
            "mfe_r_5d": 0.0,
            "mae_r_5d": 0.0,
            "mfe_r_10d": 0.0,
            "mae_r_10d": 0.0,
            "days_to_up_5d": 0,
            "days_to_up_10d": 0,
        }

    risk = atr * down_atr
    up_level = close + atr * up_atr
    down_level = close - risk
    result = {
        "outcome_valid": 1,
        "hit_up_before_down_5d": 0,
        "hit_up_before_down_10d": 0,
        "mfe_r_5d": 0.0,
        "mae_r_5d": 0.0,
        "mfe_r_10d": 0.0,
        "mae_r_10d": 0.0,
        "days_to_up_5d": 0,
        "days_to_up_10d": 0,
    }

    max_high = close
    min_low = close
    first_up_day_5 = None
    first_up_day_10 = None
    barrier_state_5 = None
    barrier_state_10 = None

    end = min(len(raw), index + 1 + horizon)
    for future_idx in range(index + 1, end):
        row = raw.iloc[future_idx]
        high = _safe_float(row["high"])
        low = _safe_float(row["low"])
        max_high = max(max_high, high)
        min_low = min(min_low, low)
        day_number = future_idx - index

        if first_up_day_10 is None and high >= up_level:
            first_up_day_10 = day_number
        if horizon == HORIZON_5 and first_up_day_5 is None and high >= up_level:
            first_up_day_5 = day_number

        if horizon >= HORIZON_5 and day_number <= HORIZON_5 and barrier_state_5 is None:
            hit_up = high >= up_level
            hit_down = low <= down_level
            if hit_up or hit_down:
                barrier_state_5 = "down" if hit_down else "up"
                if barrier_state_5 == "up":
                    first_up_day_5 = day_number

        if barrier_state_10 is None:
            hit_up = high >= up_level
            hit_down = low <= down_level
            if hit_up or hit_down:
                barrier_state_10 = "down" if hit_down else "up"
                if barrier_state_10 == "up":
                    first_up_day_10 = day_number

    max_high_5 = close
    min_low_5 = close
    for future_idx in range(index + 1, min(len(raw), index + 1 + HORIZON_5)):
        max_high_5 = max(max_high_5, _safe_float(raw.iloc[future_idx]["high"]))
        min_low_5 = min(min_low_5, _safe_float(raw.iloc[future_idx]["low"]))

    result["mfe_r_5d"] = (max_high_5 - close) / risk
    result["mae_r_5d"] = (min_low_5 - close) / risk
    result["mfe_r_10d"] = (max_high - close) / risk
    result["mae_r_10d"] = (min_low - close) / risk
    result["hit_up_before_down_5d"] = int(barrier_state_5 == "up")
    result["hit_up_before_down_10d"] = int(barrier_state_10 == "up")
    result["days_to_up_5d"] = int(first_up_day_5 or 0)
    result["days_to_up_10d"] = int(first_up_day_10 or 0)
    return result


def _build_feature_row(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    index: int,
    nifty_slice: pd.DataFrame,
    sector_series: Optional[pd.Series],
) -> dict:
    latest = features.iloc[index]
    closes = pd.to_numeric(raw["close"].iloc[: index + 1], errors="coerce")
    volumes = pd.to_numeric(raw["volume"].iloc[: index + 1], errors="coerce")
    atr = max(_safe_float(latest.get("atr_14")), 1e-9)
    close = max(_safe_float(latest.get("close_price")), 1e-9)

    returns = closes.pct_change()
    volume_mean_5 = max(_safe_float(volumes.rolling(5).mean().iloc[-1]), 1e-9)
    volume_mean_20 = max(_safe_float(volumes.rolling(20).mean().iloc[-1]), 1e-9)
    volume_std_20 = _safe_float(volumes.rolling(20).std().iloc[-1], 0.0)
    current_volume = _safe_float(volumes.iloc[-1])

    range_pct = (_safe_float(raw.iloc[index]["high"]) - _safe_float(raw.iloc[index]["low"])) / close
    gap_pct = (_safe_float(raw.iloc[index]["open"]) / max(_safe_float(raw.iloc[index - 1]["close"]), 1e-9)) - 1.0

    high_20 = max(_safe_float(v) for v in raw["high"].iloc[max(0, index - 19): index + 1])
    high_50 = max(_safe_float(v) for v in raw["high"].iloc[max(0, index - 49): index + 1])
    close_10 = closes.iloc[max(0, len(closes) - 10):]
    range_compression = (
        float(close_10.max() / max(close_10.min(), 1e-9) - 1.0)
        if len(close_10) >= 5 else 0.0
    )

    nifty_close = nifty_slice["close"] if nifty_slice is not None and not nifty_slice.empty else None
    sector_close = sector_series if sector_series is not None else None

    delivery_pct_series = pd.to_numeric(raw["delivery_percentage"].iloc[: index + 1], errors="coerce") if "delivery_percentage" in raw.columns else pd.Series(dtype=float)
    current_delivery = delivery_pct_series.iloc[-1] if len(delivery_pct_series) else np.nan
    delivery_mean_20 = delivery_pct_series.iloc[max(0, len(delivery_pct_series) - 21): -1].mean() if len(delivery_pct_series) > 1 else np.nan
    delivery_ratio = (
        float(current_delivery / delivery_mean_20)
        if pd.notna(current_delivery) and pd.notna(delivery_mean_20) and delivery_mean_20 > 0
        else 1.0
    )

    row = {
        "return_1d": _pct_change(closes, 1),
        "return_3d": _pct_change(closes, 3),
        "return_5d": _pct_change(closes, 5),
        "return_acceleration_1d": _safe_float(returns.iloc[-1] - returns.iloc[-2]) if len(returns) > 1 else 0.0,
        "return_acceleration_3d": _pct_change(closes, 3) - _pct_change(closes.iloc[:-3], 3) if len(closes) > 6 else 0.0,
        "volume_ratio_5d": current_volume / volume_mean_5,
        "volume_ratio_20d": current_volume / volume_mean_20,
        "volume_zscore_20d": (current_volume - volume_mean_20) / max(volume_std_20, 1e-9),
        "range_pct": range_pct,
        "range_atr_multiple": (range_pct * close) / atr,
        "atr_pct": atr / close,
        "atr_expansion_ratio": atr / max(_safe_float(latest.get("atr_sma_20"), atr), 1e-9),
        "gap_pct": gap_pct,
        "rsi_14": _safe_float(latest.get("rsi_14"), 50.0),
        "rsi_change_3d": _safe_float(latest.get("rsi_14"), 50.0) - _safe_float(features.iloc[max(0, index - 3)].get("rsi_14"), 50.0),
        "macd_hist_pct": _safe_float(latest.get("macd_hist")) / close,
        "macd_hist_change_pct": _safe_float(latest.get("macd_hist_diff")) / close,
        "adx_14": _safe_float(latest.get("adx_14")),
        "plus_di_minus_di": _safe_float(latest.get("plus_di")) - _safe_float(latest.get("minus_di")),
        "price_vs_ema20_pct": close / max(_safe_float(latest.get("ema_20"), close), 1e-9) - 1.0,
        "price_vs_ema50_pct": close / max(_safe_float(latest.get("ema_50"), close), 1e-9) - 1.0,
        "price_vs_ema200_pct": close / max(_safe_float(latest.get("ema_200"), close), 1e-9) - 1.0,
        "ema20_vs_ema50_pct": _safe_float(latest.get("ema_20"), close) / max(_safe_float(latest.get("ema_50"), close), 1e-9) - 1.0,
        "ema50_vs_ema200_pct": _safe_float(latest.get("ema_50"), close) / max(_safe_float(latest.get("ema_200"), close), 1e-9) - 1.0,
        "ema50_slope_20": _safe_float(latest.get("ema_50_slope_20")),
        "distance_to_20d_high_pct": close / max(high_20, 1e-9) - 1.0,
        "distance_to_50d_high_pct": close / max(high_50, 1e-9) - 1.0,
        "range_compression_10d": range_compression,
        "relative_strength_5d": _relative_strength(closes, nifty_close, 5),
        "relative_strength_20d": _relative_strength(closes, nifty_close, 20),
        "sector_strength_5d": _relative_strength(closes, sector_close, 5),
        "sector_strength_20d": _relative_strength(closes, sector_close, 20),
        "delivery_pct": _safe_float(current_delivery, 0.0),
        "delivery_pct_ratio_20d": delivery_ratio,
        "delivery_available": int(pd.notna(current_delivery)),
    }
    return row


def build_momentum_event_dataset(
    output_path: Path | str = "data/prediction/momentum_event_dataset_v1.csv",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)
    rows: list[dict] = []

    for ticker, raw_original in loader.stock_candles.items():
        if len(raw_original) < 220:
            continue
        raw = raw_original.copy()
        features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features.index = raw.index
        sector = loader.ticker_sectors.get(ticker)
        sector_series = loader.sector_series.get(sector)

        for index in range(200, len(raw) - HORIZON_10):
            as_of_date = pd.Timestamp(raw.index[index]).date()
            nifty_slice, candidate_data = loader.get_point_in_time_slice(as_of_date, min_bars_required=30)
            if ticker not in candidate_data:
                continue

            feature_row = _build_feature_row(raw, features, index, nifty_slice, sector_series)
            outcome_5 = _triple_barrier_outcome(raw, index, HORIZON_5)
            outcome_10 = _triple_barrier_outcome(raw, index, HORIZON_10)
            feature_row.update({
                "ticker_symbol": ticker,
                "trade_date": as_of_date,
                "sector": sector or "UNKNOWN",
                **outcome_5,
                "hit_up_before_down_10d": outcome_10["hit_up_before_down_10d"],
                "mfe_r_10d": outcome_10["mfe_r_10d"],
                "mae_r_10d": outcome_10["mae_r_10d"],
                "days_to_up_10d": outcome_10["days_to_up_10d"],
            })
            rows.append(feature_row)

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

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

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
    print(f"dataset_output={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/prediction/momentum_event_dataset_v1.csv"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    build_momentum_event_dataset(args.output, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
