"""Score completed daily bars for next-session momentum ignition."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.prediction.momentum_event_dataset_v2 import _vectorized_features
from src.prediction.momentum_event_dataset import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("models/momentum_ignition_v1.joblib")
DEFAULT_DATASET_PATH = Path("data/prediction/momentum_ignition_dataset_v3.csv")
MIN_HISTORY = 200


class MomentumIgnitionScorer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.features = list(artifact["features"])
        self.label = artifact["label"]
        self.model_version = artifact["model_version"]

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.features if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing scoring features: {missing}")

        scored = frame.copy()
        features = scored[self.features].apply(pd.to_numeric, errors="coerce")
        probabilities = np.asarray(self.model.predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Model predict_proba must return a 2-column probability matrix")
        scored["ignition_probability"] = probabilities[:, 1]
        scored["model_version"] = self.model_version
        return scored


def score_latest(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    latest_date = frame["trade_date"].max()
    latest = frame[frame["trade_date"] == latest_date].copy()
    scorer = MomentumIgnitionScorer(model_path)
    scored = scorer.score(latest)
    return scored.sort_values("ignition_probability", ascending=False).reset_index(drop=True)


def _benchmark_returns(loader: HistoricalDataLoader, as_of_date: date) -> tuple[float, float]:
    nifty_slice, _ = loader.get_point_in_time_slice(as_of_date, min_bars_required=30)
    if nifty_slice is None or nifty_slice.empty or "close" not in nifty_slice.columns:
        return 0.0, 0.0
    close = pd.to_numeric(nifty_slice["close"], errors="coerce")
    return float(close.pct_change(5).iloc[-1] or 0.0), float(close.pct_change(20).iloc[-1] or 0.0)


def score_latest_completed_features(model_path: Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """Score the latest completed daily bars directly from the PIT DB data.

    Unlike ``score_latest()``, this does not depend on a labeled dataset ending
    one session before the current completed bar. It builds the same feature
    pipeline used by the training dataset and uses only information available
    at the latest completed session close.
    """
    loader = HistoricalDataLoader()
    loader.load_data()
    scorer = MomentumIgnitionScorer(model_path)
    rows: list[dict] = []

    latest_date = max(loader.trading_dates)
    bench5, bench20 = _benchmark_returns(loader, latest_date)

    for ticker, raw_original in loader.stock_candles.items():
        raw = raw_original.copy()
        if len(raw) < MIN_HISTORY or latest_date not in raw.index:
            continue

        sector = loader.ticker_sectors.get(ticker)
        sector_series = loader.sector_series.get(sector)
        features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
        features.index = raw.index
        vector_features = _vectorized_features(raw, features, sector_series)
        idx = raw.index[-1]
        row = vector_features.loc[idx, FEATURE_COLUMNS].to_dict()
        row["relative_strength_5d"] = float(row["return_5d"]) - bench5
        row["relative_strength_20d"] = float(row["relative_strength_20d"]) - bench20
        row["ticker_symbol"] = ticker
        row["trade_date"] = latest_date
        rows.append(row)

    if not rows:
        raise RuntimeError("No current completed daily feature rows were available for scoring")

    frame = pd.DataFrame(rows)
    scored = scorer.score(frame)
    return scored.sort_values("ignition_probability", ascending=False).reset_index(drop=True)


def build_next_session_watchlist(
    top_n: int = 10,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Return the current completed-day model ranking for the next session."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    scored = score_latest_completed_features(model_path)
    return scored.head(top_n).copy()
