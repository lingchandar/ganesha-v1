"""Audit prediction labels against the backtest's actual next-day entry mechanics.

This script does not modify the prediction dataset. It reconstructs each historical
setup, applies the same next-day buy-limit fill rules used by the backtest, and then
compares the existing signal-day label with a fill-aware label using the same daily
exit ordering as PrecisionExitEngine.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional

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
    """Match BacktestEngine._execute_pending_orders fill rules."""
    dates = pd.to_datetime(raw.index).date
    matches = [i for i, d in enumerate(dates) if d > signal_date]
    if not matches:
        return None

    i = matches[0]
    candle = raw.iloc[i]
    c_open = float(candle["open"])
    c_low = float(candle["low"])
    c_high = float(candle["high"])

    if c_open <= entry_price:
        return i, c_open
    if c_low <= entry_price <= c_high:
        return i, entry_price
    return None


def _fill_aware_label(
    raw: pd.DataFrame,
    fill_idx: int,
    fill_price: float,
    signal,
    horizon: int,
) -> Optional[int]:
    """Evaluate outcome using the actual fill price and real exit ordering.

    The backtest does not evaluate a newly opened position on its fill day, so
    outcome evaluation starts on the following trading session.
    """
    stop = float(signal.stop_loss)
    target = float(signal.target_price)
    invalidation = float(signal.invalidation_level)
    if fill_price <= stop:
        return None

    risk = fill_price - stop
    if risk <= 0:
        return None

    future_end = min(fill_idx + 1 + horizon, len(raw))
    for idx in range(fill_idx + 1, future_end):
        row = raw.iloc[idx]
        result = PrecisionExitEngine.evaluate_active_trade(
            entry_price=fill_price,
            stop_loss=stop,
            target_price=target,
            invalidation_level=invalidation,
            days_held=idx - fill_idx,
            daily_candle={
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            },
            has_upcoming_earnings=False,
        )
        if result.is_exit_triggered:
            return 1 if result.exit_reason == "TARGET_HIT (+2R)" else 0

    return 0


def audit(dataset_path: Path, start_date: Optional[date], end_date: Optional[date]) -> None:
    frame = pd.read_csv(dataset_path)
    required = {
        "ticker_symbol",
        "trade_date",
        "setup_type",
        "label_target_before_stop",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    if start_date:
        frame = frame[frame["trade_date"] >= start_date]
    if end_date:
        frame = frame[frame["trade_date"] <= end_date]

    print(f"dataset_rows={len(frame)}")
    print(f"dataset_positive_rate={frame['label_target_before_stop'].mean():.4f}")

    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)

    rows = []
    for ticker, raw in loader.stock_candles.items():
        ticker_rows = frame[frame["ticker_symbol"] == ticker]
        if ticker_rows.empty or len(raw) < 160:
            continue

        features_all = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features_all.index = raw.index
        date_to_idx = {pd.Timestamp(d).date(): i for i, d in enumerate(raw.index)}

        for _, csv_row in ticker_rows.iterrows():
            trade_date = csv_row["trade_date"]
            i = date_to_idx.get(trade_date)
            if i is None or i < 120:
                rows.append({"status": "MISSING_SIGNAL_DATE"})
                continue

            window = features_all.iloc[: i + 1]
            signals = _candidate_signals(ticker, window)
            signal = next(
                (s for s in signals if s.setup_type == csv_row["setup_type"]),
                None,
            )
            if signal is None:
                rows.append({"status": "SIGNAL_NOT_REPRODUCED"})
                continue

            fill = _next_day_fill(raw, trade_date, float(signal.entry_price))
            if fill is None:
                rows.append({
                    "status": "NO_NEXT_DAY_FILL",
                    "old_label": int(csv_row["label_target_before_stop"]),
                    "setup_type": signal.setup_type,
                })
                continue

            fill_idx, fill_price = fill
            new_label = _fill_aware_label(
                raw,
                fill_idx,
                fill_price,
                signal,
                horizon=15,
            )
            if new_label is None:
                rows.append({"status": "INVALID_FILL_RISK"})
                continue

            old_label = int(csv_row["label_target_before_stop"])
            rows.append({
                "status": "AUDITED",
                "old_label": old_label,
                "new_label": new_label,
                "label_changed": old_label != new_label,
                "setup_type": signal.setup_type,
                "fill_improved_pct": round((fill_price / float(signal.entry_price) - 1.0) * 100.0, 4),
            })

    audit_frame = pd.DataFrame(rows)
    if audit_frame.empty:
        raise RuntimeError("No audit results were produced")

    print("\nstatus_counts:")
    print(audit_frame["status"].value_counts().to_string())

    audited = audit_frame[audit_frame["status"] == "AUDITED"]
    print(f"\naudited_rows={len(audited)}")
    if not audited.empty:
        print(f"old_positive_rate={audited['old_label'].mean():.4f}")
        print(f"fill_aware_positive_rate={audited['new_label'].mean():.4f}")
        print(f"label_changed_rows={int(audited['label_changed'].sum())}")
        print(f"label_change_rate={audited['label_changed'].mean():.4f}")
        print(f"avg_fill_improvement_pct={audited['fill_improved_pct'].mean():.4f}")
        print("\nlabel_confusion:")
        print(
            pd.crosstab(
                audited["old_label"],
                audited["new_label"],
                rownames=["existing_label"],
                colnames=["fill_aware_label"],
            ).to_string()
        )
        print("\nlabel_changes_by_setup:")
        print(
            audited.groupby("setup_type")["label_changed"]
            .agg(["count", "sum", "mean"])
            .sort_values("sum", ascending=False)
            .to_string()
        )

    output = dataset_path.with_name("prediction_label_audit.csv")
    audit_frame.to_csv(output, index=False)
    print(f"\naudit_output={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/prediction/entry_dataset.csv"),
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    audit(args.dataset, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
