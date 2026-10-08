"""Historical point-in-time dataset builder for entry prediction."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.setups.swing_setups import SwingSetupScanner
from src.prediction.engine import PredictionFeatureBuilder


def _candidate_signals(ticker: str, features: pd.DataFrame):
    return [
        s for s in (
            SwingSetupScanner._check_classic_breakout(ticker, features, False),
            SwingSetupScanner._check_breakout_retest(ticker, features, False),
            SwingSetupScanner._check_trend_pullback(ticker, features),
            SwingSetupScanner._check_demand_bounce(ticker, features),
            SwingSetupScanner._check_continuation_flag(ticker, features),
            SwingSetupScanner._check_rs_leader_breakout(ticker, features),
        ) if s is not None
    ]


def _label_outcome(
    features: pd.DataFrame,
    entry_idx: int,
    signal,
    horizon: int = 15,
) -> Optional[int]:
    """Return 1 when target is reached before stop, otherwise 0.

    The future window starts strictly after the signal candle. This keeps the
    label point-in-time safe and avoids using the entry candle to determine
    the outcome.
    """
    risk = signal.entry_price - signal.stop_loss
    if risk <= 0:
        return None

    future = features.iloc[entry_idx + 1 : entry_idx + 1 + horizon]
    if future.empty:
        return None

    for _, row in future.iterrows():
        if float(row["low_price"]) <= signal.stop_loss:
            return 0
        if float(row["high_price"]) >= signal.target_price:
            return 1
    return 0


def build_dataset(
    output_path: Path | str = "data/prediction/entry_dataset.csv",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)

    rows: List[dict] = []
    candidate_count = 0
    labeled_count = 0

    for ticker, raw in loader.stock_candles.items():
        if len(raw) < 160:
            continue

        features_all = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features_all.index = raw.index

        for i in range(120, len(features_all) - 15):
            window = features_all.iloc[: i + 1]
            signals = _candidate_signals(ticker, window)
            candidate_count += len(signals)

            for signal in signals:
                label = _label_outcome(features_all, i, signal, horizon=15)
                if label is None:
                    continue

                feature_row = PredictionFeatureBuilder.build(signal, window).iloc[0].to_dict()
                feature_row.update({
                    "ticker_symbol": ticker,
                    "trade_date": window.index[-1],
                    "setup_type": signal.setup_type,
                    "label_target_before_stop": int(label),
                })
                rows.append(feature_row)
                labeled_count += 1

    if not rows:
        raise RuntimeError(
            "Prediction dataset builder produced zero labeled examples "
            f"(candidate_signals={candidate_count}, labeled={labeled_count})"
        )

    frame = pd.DataFrame(rows).sort_values(
        ["trade_date", "ticker_symbol", "setup_type"]
    ).reset_index(drop=True)

    feature_columns = PredictionFeatureBuilder.FEATURE_COLUMNS
    if frame[feature_columns].isna().any().any():
        raise ValueError("Prediction dataset contains NaN feature values")

    duplicate_keys = frame.duplicated(
        subset=["ticker_symbol", "trade_date", "setup_type"],
        keep=False,
    )
    if duplicate_keys.any():
        raise ValueError(
            "Prediction dataset contains duplicate ticker/date/setup examples"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    positive = int(frame["label_target_before_stop"].sum())
    negative = len(frame) - positive
    positive_rate = positive / len(frame)
    print(
        "prediction_dataset="
        f"rows={len(frame)} candidates={candidate_count} "
        f"positive={positive} negative={negative} "
        f"positive_rate={positive_rate:.4f} "
        f"tickers={frame['ticker_symbol'].nunique()} "
        f"start={frame['trade_date'].min()} end={frame['trade_date'].max()}"
    )
    print("prediction_dataset_by_setup:")
    print(
        frame.groupby("setup_type")["label_target_before_stop"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .to_string()
    return output
