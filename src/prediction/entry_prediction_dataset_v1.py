"""Point-in-time entry prediction dataset with explicit R-multiple outcomes."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.prediction.dataset import _candidate_signals
from src.prediction.engine import PredictionFeatureBuilder

HORIZON_SESSIONS = 15


def _label_trade(features: pd.DataFrame, entry_idx: int, signal, horizon: int) -> Optional[dict]:
    entry = float(signal.entry_price)
    stop = float(signal.stop_loss)
    target = float(signal.target_price)
    risk = entry - stop
    if risk <= 0 or target <= entry:
        return None

    future = features.iloc[entry_idx + 1 : entry_idx + 1 + horizon]
    if future.empty:
        return None

    mfe_r = 0.0
    mae_r = 0.0
    target_hit = False
    stop_hit = False
    days_to_target = None
    days_to_stop = None

    for offset, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high_price"])
        low = float(row["low_price"])
        mfe_r = max(mfe_r, (high - entry) / risk)
        mae_r = min(mae_r, (low - entry) / risk)

        # Conservative same-bar handling: adverse barrier first.
        if not stop_hit and low <= stop:
            stop_hit = True
            days_to_stop = offset
        if not target_hit and high >= target:
            target_hit = True
            days_to_target = offset

        if stop_hit or target_hit:
            break

    if stop_hit and target_hit:
        target_before_stop = int(days_to_target < days_to_stop)
    elif target_hit:
        target_before_stop = 1
    elif stop_hit:
        target_before_stop = 0
    else:
        target_before_stop = 0

    return {
        "label_target_before_stop": target_before_stop,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "days_to_target": days_to_target if days_to_target is not None else 0,
        "days_to_stop": days_to_stop if days_to_stop is not None else 0,
        "outcome_valid": int(target_hit or stop_hit),
    }


def build_dataset(
    output_path: Path | str = "data/prediction/entry_prediction_dataset_v1.csv",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    horizon: int = HORIZON_SESSIONS,
) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)

    rows: List[dict] = []
    for ticker, raw in loader.stock_candles.items():
        if len(raw) < 160:
            continue

        features_all = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features_all.index = raw.index

        for i in range(120, len(features_all) - horizon):
            window = features_all.iloc[: i + 1]
            for signal in _candidate_signals(ticker, window):
                outcome = _label_trade(features_all, i, signal, horizon)
                if outcome is None:
                    continue

                feature_row = PredictionFeatureBuilder.build(signal, window).iloc[0].to_dict()
                feature_row.update(
                    {
                        "ticker_symbol": ticker,
                        "trade_date": window.index[-1],
                        "setup_type": signal.setup_type,
                        "entry_price": float(signal.entry_price),
                        "stop_loss": float(signal.stop_loss),
                        "target_price": float(signal.target_price),
                        "risk_points": float(signal.entry_price - signal.stop_loss),
                        **outcome,
                    }
                )
                rows.append(feature_row)

    if not rows:
        raise RuntimeError("Entry prediction v1 dataset produced zero labeled examples")

    frame = pd.DataFrame(rows).sort_values(
        ["trade_date", "ticker_symbol", "setup_type"]
    ).reset_index(drop=True)

    required = PredictionFeatureBuilder.FEATURE_COLUMNS + [
        "ticker_symbol",
        "trade_date",
        "setup_type",
        "label_target_before_stop",
        "mfe_r",
        "mae_r",
        "outcome_valid",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Entry prediction v1 dataset missing columns: {missing}")
    if frame[PredictionFeatureBuilder.FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Entry prediction v1 dataset contains NaN features")
    if frame.duplicated(["ticker_symbol", "trade_date", "setup_type"]).any():
        raise ValueError("Entry prediction v1 dataset contains duplicate examples")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    valid = frame[frame["outcome_valid"] == 1]
    print(
        "entry_prediction_v1_dataset="
        f"rows={len(frame)} valid_outcomes={len(valid)} "
        f"positive={int(frame['label_target_before_stop'].sum())} "
        f"positive_rate={frame['label_target_before_stop'].mean():.4f} "
        f"tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()}"
    )
    print("entry_prediction_v1_by_setup:")
    print(
        frame.groupby("setup_type")["label_target_before_stop"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .to_string()
    )
    return output


if __name__ == "__main__":
    build_dataset()
