"""Build an entry-prediction dataset aligned with real next-day backtest fills."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.ingestion.expiry_calendar import NSEDerivativeExpiryFilter
from src.prediction.engine import PredictionFeatureBuilder
from src.prediction.trade_eligibility import evaluate_signal_eligibility
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


def _candidate_signals(ticker: str, window: pd.DataFrame, is_expiry_week: bool):
    """Match the six core setup methods used by SwingSetupScanner."""
    signals = []
    for method in SETUP_METHODS:
        if method in (
            SwingSetupScanner._check_classic_breakout,
            SwingSetupScanner._check_breakout_retest,
        ):
            signal = method(ticker, window, is_expiry_week)
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

    return {
        "label_target_before_stop": int(exit_label),
        "mfe_r_15": round(float((max_high - fill_price) / risk), 4),
        "mae_r_15": round(float((min_low - fill_price) / risk), 4),
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
    expiry_filter = NSEDerivativeExpiryFilter()

    rows: List[dict] = []
    audit_rows: List[dict] = []
    candidate_count = 0
    top_signal_count = 0
    eligible_signal_count = 0
    fill_count = 0
    invalid_fill_count = 0
    rejected_by_gate: dict[str, int] = {}

    for ticker, raw in loader.stock_candles.items():
        if len(raw) < 160:
            continue

        features_all = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features_all.index = raw.index

        for i in range(120, len(features_all) - 15):
            signal_date = pd.Timestamp(features_all.index[i]).date()
            window = features_all.iloc[: i + 1]

            nifty_slice, candidate_data = loader.get_point_in_time_slice(
                as_of_date=signal_date,
                min_bars_required=30,
            )
            ticker_data = candidate_data.get(ticker)
            if ticker_data is None:
                continue
            ticker_data = dict(ticker_data)
            ticker_data["ticker_symbol"] = ticker

            expiry_week = expiry_filter.is_expiry_week_active(signal_date).get(
                "is_expiry_week", False
            )
            signals = _candidate_signals(ticker, window, expiry_week)
            candidate_count += len(signals)
            if not signals:
                continue

            # BacktestEngine uses signals[0] as the top signal for a stock/day.
            signal = signals[0]
            top_signal_count += 1

            eligibility = evaluate_signal_eligibility(
                signal=signal,
                candidate_data=ticker_data,
                nifty_slice=nifty_slice,
                is_expiry_week=expiry_week,
                min_reward_risk=2.0,
                enable_regime_filter=True,
                enable_rs_filter=True,
            )
            if not eligibility.eligible:
                rejected_by_gate[eligibility.reason] = rejected_by_gate.get(eligibility.reason, 0) + 1
                audit_rows.append({
                    "ticker_symbol": ticker,
                    "signal_date": signal_date,
                    "setup_type": signal.setup_type,
                    "status": eligibility.reason,
                    "signal_entry_price": float(signal.entry_price),
                    "risk_reward_ratio": eligibility.risk_reward_ratio,
                })
                continue

            eligible_signal_count += 1
            fill = _next_day_fill(raw, signal_date, float(signal.entry_price))
            if fill is None:
                audit_rows.append({
                    "ticker_symbol": ticker,
                    "signal_date": signal_date,
                    "setup_type": signal.setup_type,
                    "status": "NO_NEXT_DAY_FILL",
                    "signal_entry_price": float(signal.entry_price),
                    "risk_reward_ratio": eligibility.risk_reward_ratio,
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
                    "risk_reward_ratio": eligibility.risk_reward_ratio,
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
                "risk_reward_ratio_backtest": round(float(eligibility.risk_reward_ratio), 4),
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
                "risk_reward_ratio": eligibility.risk_reward_ratio,
                "label_target_before_stop": outcome["label_target_before_stop"],
            })

    if not rows:
        raise RuntimeError("Trade-eligible prediction dataset produced zero training rows")

    frame = pd.DataFrame(rows).sort_values(
        ["trade_date", "ticker_symbol", "setup_type"]
    ).reset_index(drop=True)

    feature_columns = PredictionFeatureBuilder.FEATURE_COLUMNS
    if frame[feature_columns].isna().any().any():
        raise ValueError("Trade-eligible prediction dataset contains NaN feature values")

    duplicate_keys = frame.duplicated(
        subset=["ticker_symbol", "trade_date"],
        keep=False,
    )
    if duplicate_keys.any():
        raise ValueError("Trade-eligible dataset contains duplicate ticker/date examples")

    output = Path(output_path)
    audit_output = Path(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_output, index=False)

    positive = int(frame["label_target_before_stop"].sum())
    print(
        "trade_eligible_prediction_dataset="
        f"rows={len(frame)} candidates={candidate_count} "
        f"top_signals={top_signal_count} eligible_signals={eligible_signal_count} "
        f"fills={fill_count} invalid_fills={invalid_fill_count} positive={positive} "
        f"negative={len(frame) - positive} positive_rate={positive / len(frame):.4f} "
        f"tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()}"
    )
    print("rejected_by_gate:")
    print(pd.Series(rejected_by_gate).sort_values(ascending=False).to_string() if rejected_by_gate else "none")
    print("trade_eligible_dataset_by_setup:")
    print(
        frame.groupby("setup_type")["label_target_before_stop"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .to_string()
    )
    print(f"dataset_output={output}")
    print(f"audit_output={audit_output}")
    return output, audit_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/prediction/entry_dataset_v5.csv"))
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/prediction/prediction_fill_aware_audit_v5.csv"),
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    build_fill_aware_dataset(
        output_path=args.output,
        audit_path=args.audit_output,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
