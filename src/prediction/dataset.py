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


def _label_outcome(df: pd.DataFrame, signal, horizon: int = 15) -> Optional[int]:
    """1 iff +2R is reached before the structural stop; otherwise 0."""
    risk = signal.entry_price - signal.stop_loss
    if risk <= 0:
        return None
    future = df.iloc[df.index.get_loc(df.index[-1]) + 1 : df.index.get_loc(df.index[-1]) + 1 + horizon]
    if future.empty:
        return None
    for _, row in future.iterrows():
        if float(row["low_price"]) <= signal.stop_loss:
            return 0
        if float(row["high_price"]) >= signal.target_price:
            return 1
    return 0


def build_dataset(output_path: Path | str = "data/prediction/entry_dataset.csv", start_date: Optional[date] = None, end_date: Optional[date] = None) -> Path:
    loader = HistoricalDataLoader()
    loader.load_data(start_date=start_date, end_date=end_date)
    rows: List[dict] = []
    for ticker, raw in loader.stock_candles.items():
        if len(raw) < 160:
            continue
        features_all = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features_all.index = raw.index
        for i in range(120, len(features_all) - 15):
            window = features_all.iloc[: i + 1]
            signals = _candidate_signals(ticker, window)
            for signal in signals:
                label = _label_outcome(features_all.iloc[: i + 16], signal, horizon=15)
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
    if not rows:
        raise RuntimeError("Prediction dataset builder produced zero labeled examples")
    frame = pd.DataFrame(rows).sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output
