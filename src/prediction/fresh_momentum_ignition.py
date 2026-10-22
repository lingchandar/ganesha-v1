"""Build the next-session momentum watchlist from fresh FYERS daily candles."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.data_loader import HistoricalDataLoader
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.ingestion.fyers_auth import FyersAuthManager
from src.ingestion.market_data_client import MarketDataClient
from src.prediction.momentum_event_dataset import FEATURE_COLUMNS
from src.prediction.momentum_event_dataset_v2 import _vectorized_features
from src.prediction.momentum_ignition_scorer import MomentumIgnitionScorer

IST = ZoneInfo("Asia/Kolkata")
MIN_HISTORY = 200
# FYERS history rejects requests that exceed its supported historical range.
# 300 calendar days still provides >200 NSE sessions in normal conditions.
LOOKBACK_DAYS = 300
FRESH_REQUEST_INTERVAL_SECONDS = 1.05
FYERS_NIFTY50 = "NSE:NIFTY50-INDEX"


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.rename(
        columns={
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "volume_traded": "volume",
        }
    ).copy()
    out["trade_date"] = pd.to_datetime(out["timestamp"]).dt.date
    out = out.drop_duplicates("trade_date").sort_values("trade_date").set_index("trade_date")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close", "volume"])


def _latest_completed_date() -> date:
    now = datetime.now(IST)
    if now.weekday() < 5 and (now.hour, now.minute) >= (15, 30):
        return now.date()
    return (now - timedelta(days=1)).date()


def _fetch_daily(
    client: MarketDataClient,
    symbol: str,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    """Fetch one daily series while spacing requests to avoid FYERS 429s."""
    sleep(FRESH_REQUEST_INTERVAL_SECONDS)
    return _normalise(client.fetch_historical_candles(symbol, from_date, to_date, "D"))


def build_fresh_next_session_watchlist(
    top_n: int = 10,
    model_path: Path = Path("models/momentum_ignition_v1.joblib"),
) -> pd.DataFrame:
    """Fetch fresh daily OHLCV from FYERS and score the latest completed session."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if not FyersAuthManager.load_cached_token():
        raise RuntimeError(
            "No valid FYERS access token. Run 'PYTHONPATH=. .venv/bin/python -m src.ingestion.fyers_auth' first."
        )

    loader = HistoricalDataLoader()
    loader.load_data()
    completed_date = _latest_completed_date()
    from_date = completed_date - timedelta(days=LOOKBACK_DAYS)
    client = MarketDataClient()
    if not client.authenticate():
        raise RuntimeError("FYERS authentication failed while refreshing daily data")

    try:
        nifty = _fetch_daily(client, FYERS_NIFTY50, from_date, completed_date)
        nifty_close = pd.to_numeric(nifty.get("close", pd.Series(dtype=float)), errors="coerce")
        bench5 = float(nifty_close.pct_change(5).iloc[-1]) if len(nifty_close) >= 6 else 0.0
        bench20 = float(nifty_close.pct_change(20).iloc[-1]) if len(nifty_close) >= 21 else 0.0

        scorer = MomentumIgnitionScorer(model_path)
        rows: list[dict] = []
        for ticker in loader.universe_df["ticker_symbol"].tolist():
            raw = _fetch_daily(client, ticker, from_date, completed_date)
            if len(raw) < MIN_HISTORY:
                continue

            # Delivery is not available from this FYERS endpoint. Keep the
            # feature neutral rather than inventing current-day delivery data.
            raw["delivery_percentage"] = np.nan
            sector = loader.ticker_sectors.get(ticker)
            sector_series = loader.sector_series.get(sector)
            features = TechnicalAnalysisEngine.compute_all_features(raw.reset_index())
            features.index = raw.index
            vector = _vectorized_features(raw, features, sector_series)
            idx = vector.index[-1]
            row = vector.loc[idx, FEATURE_COLUMNS].to_dict()
            row["relative_strength_5d"] = float(row["return_5d"]) - bench5
            row["relative_strength_20d"] = float(row["return_20d"]) - bench20
            row["ticker_symbol"] = ticker
            row["trade_date"] = idx
            rows.append(row)

        if not rows:
            raise RuntimeError("FYERS returned no usable daily history for the controlled universe")
        scored = scorer.score(pd.DataFrame(rows))
        return scored.sort_values("ignition_probability", ascending=False).head(top_n).reset_index(drop=True)
    finally:
        client.close()
