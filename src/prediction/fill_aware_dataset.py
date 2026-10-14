"""Build an entry-prediction dataset aligned with real next-day backtest fills."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.prediction.engine import PredictionFeatureBuilder
from src.risk.exit_engine import PrecisionExitEngine
from src.setups.swing_setups import SwingSetupScanner


SETUP_METHODS = (
    SwingSetupScanner._check_classic_breakout,
    SwingSetupScanner._check_breakout_retest,
    SwingSetupScanner._check_trend_pullback,
    SwingSetupScanner._check_demand_bounce,
    SwingSetupScanner._check_continuation_flag,
    SwingSetupScanner._check_rs_leader_breakout,
)


def _candidate_signals(ticker: str, window: pd.DataFrame):
    signals = []
    for method in SETUP_METHODS:
        if method in (
            SwingSetupScanner._check_classic_breakout,
            SwingSetupScanner._check_breakout_retest,
        ):
            signal = method(ticker, window, False)
        else:
            signal = method(ticker, window)
        if signal is not None:
            signals.append(signal)
    return signals


def _next_day_fill(raw: pd.DataFrame, signal_date: date, entry_price: float):
    """Match BacktestEngine's one-session pending-limit fill rules."""
    dates = pd.to_datetime(raw.index).date
    matches = [i for i, d in enumerate(dates) if d > signal_date]
    if not matches:
        return None

    fill_idx = matches[0]
    candle = raw.iloc[fill_idx]
    open_price = float(candle["open"])
    low_price = float(candle["low"])
    high_price = float(candle["high"])

    if open_price <= entry_price:
        return fill_idx, open_price
    if low_price <= entry_price <= high_price:
        return fill_idx, entry_price
    return None


def _evaluate_trade(
    raw: pd.DataFrame,
    fill_idx: int,
    fill_price: float,
    signal,
    horizon: int,
) -> Optional[dict]:
    """Return fill-aware outcome metrics using the centralized exit ordering."""
    stop = float(signal.stop_loss)
    target = float(signal.target_price)
    invalidation = float(signal.invalidation_level)
    risk = fill_price - stop
    if risk <= 0:
        return None

    future_end = min(fill_idx + 1 + horizon, len(raw))
    if fill_idx + 1 >= future_end:
        return None

    max_high = float("-inf")
    min_low = float("inf")
    exit_label = 0
    holding_days = 0

    for idx in range(fill_idx + 1, future_end):
        row = raw.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        max_high = max(max_high, high)
        min_low = min(min_low, low)
        holding_days = idx - fill_idx

        result = PrecisionExitEngine.evaluate_active_trade(
            entry_price=fill_price,
            stop_loss=stop,
            target_price=target,
            invalidation_level=invalidation,
            days_held=holding_days,
            daily_candle={
                "open": float(row["open"]),
                "high": high,
                "low": low,
                "close": float(row["close"]),
            },
            has_upcoming_earnings=False,
        )
        if result.is_exit_triggered:
            exit_label = 1 if result.exit_reason == "TARGET_HIT (+2R)" else 0
            break

    if max_high == float("-inf") or min_low == float("inf"):
        return None

    mfe_r = (max_high - fill_price) / risk
    mae_r = (min_low - fill_price) / risk
    return {
        "label_target_before_stop": int(exit_label),
        "mfe_r_15": round(float(mfe_r), 4),
        "mae_r_15": round(float(mae_r), 4),
        "holding_days_15": int(holding_days),
    }


def build_fill_aware_dataset(
    output_path: Path | str = "data/prediction/entry_dataset_v5.csv",
    audit_path: Path | str = "data/prediction/prediction_fill_aware_audit_v5.csv",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> tuple[Path, Path]:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)

    rows: List[dict] = []
    audit_rows: List[dict] = []
    candidate_count = 0
    top_signal_count = 0
    fill_count = 0
    invalid_fill_count = 0

    for ticker, raw in loader.stock_candles.items():
        if len(raw) < 160:
            continue

        features_all = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features_all.index = raw.index

        for i in range(120, len(features_all) - 15):
            signal_date = pd.Timestamp(features_all.index[i]).date()
            window = features_all.iloc[: i + 1]
            signals = _candidate_signals(ticker, window)
            candidate_count += len(signals)
            if not signals:
                continue

            # BacktestEngine uses signals[0] as the top signal for a stock/day.
            signal = signals[0]
            top_signal_count += 1

            fill = _next_day_fill(raw, signal_date, float(signal.entry_price))
            if fill is None:
                audit_rows.append({
                    "ticker_symbol": ticker,
                    "signal_date": signal_date,
                    "setup_type": signal.setup_type,
                    "status": "NO_NEXT_DAY_FILL",
                    "signal_entry_price": float(signal.entry_price),
                })
                continue

            fill_idx, fill_price = fill
            fill_count += 1
            outcome = _evaluate_trade(raw, fill_idx, fill_price, signal, horizon=15)
            if outcome is None:
                invalid_fill_count += 1
                audit_rows.append({
                    "ticker_symbol": ticker,
                    "signal_date": signal_date,
                    "setup_type": signal.setup_type,
                    "status": "INVALID_FILL_RISK",
                    "signal_entry_price": float(signal.entry_price),
                    "actual_fill_price": float(fill_price),
                })
                continue

            feature_row = PredictionFeatureBuilder.build(signal, window).iloc[0].to_dict()
            feature_row.update({
                "ticker_symbol": ticker,
                "trade_date": signal_date,
                "entry_date": pd.Timestamp(raw.index[fill_idx]).date(),
                "setup_type": signal.setup_type,
                "signal_entry_price": round(float(signal.entry_price), 4),
                "actual_fill_price": round(float(fill_price), 4),
                "fill_improvement_pct": round(
                    (float(fill_price) / float(signal.entry_price) - 1.0) * 100.0,
                    4,
                ),
                "stop_loss": round(float(signal.stop_loss), 4),
                "target_price": round(float(signal.target_price), 4),
                **outcome,
            })
            rows.append(feature_row)
            audit_rows.append({
                "ticker_symbol": ticker,
                "signal_date": signal_date,
                "setup_type": signal.setup_type,
                "status": "FILLED_AND_LABELED",
                "signal_entry_price": float(signal.entry_price),
                "actual_fill_price": float(fill_price),
                "label_target_before_stop": outcome["label_target_before_stop"],
            })

    if not rows:
        raise RuntimeError("Fill-aware prediction dataset produced zero training rows")

    frame = pd.DataFrame(rows).sort_values(
        ["trade_date", "ticker_symbol", "setup_type"]
    ).reset_index(drop=True)

    feature_columns = PredictionFeatureBuilder.FEATURE_COLUMNS
    if frame[feature_columns].isna().any().any():
        raise ValueError("Fill-aware prediction dataset contains NaN feature values")

    duplicate_keys = frame.duplicated(
        subset=["ticker_symbol", "trade_date"],
        keep=False,
    )
    if duplicate_keys.any():
        raise ValueError("Fill-aware dataset contains duplicate ticker/date examples")

    output = Path(output_path)
    audit_output = Path(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_output, index=False)

    positive = int(frame["label_target_before_stop"].sum())
    print(
        "fill_aware_prediction_dataset="
        f"rows={len(frame)} candidates={candidate_count} "
        f"top_signals={top_signal_count} fills={fill_count} "
        f"invalid_fills={invalid_fill_count} positive={positive} "
        f"negative={len(frame) - positive} "
        f"positive_rate={positive / len(frame):.4f} "
        f"tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()}"
    )
    print("fill_aware_dataset_by_setup:")
    print(
        frame.groupby("setup_type")["label_target_before_stop"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .to_string()
    )
    print(f"dataset_output={output}")
    print(f"audit_output={audit_output}")
    return output, audit_output
