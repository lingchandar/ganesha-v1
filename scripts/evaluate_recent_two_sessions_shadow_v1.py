"""Classify the two most recent completed sessions for the shadow model.

This is a diagnostic-only recent-session check. It does not promote the model,
change EntryGate, place orders, or claim a completed trade outcome when the
15-session forward outcome window is not yet available.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from scripts.trend_pullback_validation_universe_v1 import TREND_PULLBACK_VALIDATION_SYMBOLS
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.ingestion.market_data_client import MarketDataClient
from src.prediction.dataset import _candidate_signals
from src.prediction.engine import PredictionFeatureBuilder

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
DEFAULT_OUTPUT = Path("data/prediction/recent_two_sessions_shadow_v1.csv")
SETUP_TYPE = "TREND_PULLBACK"


def _value(signal: Any, name: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(name, default)
    return getattr(signal, name, default)


def load_model(path: Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("model_version") != "entry_prediction_extra_trees_candidate_v1":
        raise ValueError("Unsupported candidate model version")
    if bundle.get("production_ready") is not False or bundle.get("promotion_status") != "candidate_only":
        raise ValueError("Candidate model metadata is not safely candidate-only")
    if list(bundle.get("feature_columns", [])) != list(PredictionFeatureBuilder.FEATURE_COLUMNS):
        raise ValueError("Candidate feature schema does not match PredictionFeatureBuilder")
    return bundle


def recent_completed_sessions() -> tuple[date, date]:
    """Return the latest two NSE sessions available before the current day."""
    # We intentionally fetch a wider calendar window and derive the actual
    # completed dates from FYERS daily candles, rather than assuming weekdays.
    end = datetime.now(IST).date()
    return end - timedelta(days=2), end - timedelta(days=1)


def evaluate_symbol(symbol: str, target_dates: set[date], bundle: dict) -> list[dict]:
    end_date = datetime.now(IST).date()
    start_date = end_date - timedelta(days=60)

    with MarketDataClient() as client:
        if not client.authenticate():
            raise RuntimeError("FYERS authentication is unavailable")
        raw = client.fetch_historical_candles(symbol, start_date, end_date, resolution="D")

    if raw.empty:
        return []

    raw = raw.reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    if raw["timestamp"].dt.tz is None:
        raw["timestamp"] = raw["timestamp"].dt.tz_localize(IST)
    else:
        raw["timestamp"] = raw["timestamp"].dt.tz_convert(IST)
    raw["session_date"] = raw["timestamp"].dt.date

    features = TechnicalAnalysisEngine.compute_all_features(raw.copy())
    features["timestamp"] = raw["timestamp"].values
    features["session_date"] = raw["session_date"].values

    rows: list[dict] = []
    for i in range(len(features)):
        session_date = features.iloc[i]["session_date"]
        if session_date not in target_dates:
            continue

        window = features.iloc[: i + 1]
        signals = [
            signal
            for signal in _candidate_signals(symbol, window)
            if str(_value(signal, "setup_type", "")) == SETUP_TYPE
        ]

        if not signals:
            rows.append({
                "ticker_symbol": symbol,
                "trade_date": session_date,
                "classification": "NO_SETUP",
                "setup_type": SETUP_TYPE,
                "probability_target_before_stop": np.nan,
                "confidence": np.nan,
                "entry_price": np.nan,
                "stop_loss": np.nan,
                "target_price": np.nan,
                "outcome_status": "PENDING_FORWARD_VALIDATION",
            })
            continue

        for signal in signals:
            feature_row = PredictionFeatureBuilder.build(signal, window).reindex(
                columns=bundle["feature_columns"], fill_value=0.0
            )
            probability = float(bundle["model"].predict_proba(feature_row)[0, 1])
            confidence = abs(probability - 0.5) * 2.0

            if probability >= 0.30:
                classification = "HIGH_CONFIDENCE_CANDIDATE"
            elif probability >= 0.20:
                classification = "ENTRY_CANDIDATE"
            else:
                classification = "SETUP_QUALIFIED_LOW_PROBABILITY"

            rows.append({
                "ticker_symbol": symbol,
                "trade_date": session_date,
                "classification": classification,
                "setup_type": SETUP_TYPE,
                "probability_target_before_stop": probability,
                "confidence": confidence,
                "entry_price": float(_value(signal, "entry_price", np.nan)),
                "stop_loss": float(_value(signal, "stop_loss", np.nan)),
                "target_price": float(_value(signal, "target_price", np.nan)),
                "outcome_status": "PENDING_FORWARD_VALIDATION",
            })

    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", action="append", dest="dates", help="Completed session date YYYY-MM-DD; repeat twice.")
    parser.add_argument("--symbols", nargs="+", default=list(TREND_PULLBACK_VALIDATION_SYMBOLS))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.dates:
        target_dates = {date.fromisoformat(value) for value in args.dates}
    else:
        target_dates = set(recent_completed_sessions())

    if len(target_dates) != 2:
        raise ValueError("Exactly two distinct completed session dates are required")

    bundle = load_model(args.model)
    print("=== GANESHA V1 RECENT TWO-SESSION SHADOW CLASSIFICATION ===")
    print(f"DATES: {', '.join(str(value) for value in sorted(target_dates))}")
    print(f"UNIVERSE SYMBOLS: {len(args.symbols)}")
    print("MODEL: entry_prediction_extra_trees_candidate_v1 | STATUS: candidate_only")

    rows: list[dict] = []
    for symbol in args.symbols:
        rows.extend(evaluate_symbol(symbol, target_dates, bundle))

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("RESULT: no daily records returned for the requested sessions.")
        return 2

    frame = frame.sort_values(["trade_date", "classification", "ticker_symbol"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print("\n=== CLASSIFICATION BREAKDOWN ===")
    print(frame.groupby(["trade_date", "classification"]).size().to_string())

    candidates = frame[frame["classification"].isin({"ENTRY_CANDIDATE", "HIGH_CONFIDENCE_CANDIDATE"})]
    print("\n=== CANDIDATES ===")
    if candidates.empty:
        print("No entry candidates in the two requested sessions.")
    else:
        print(candidates[[
            "trade_date", "ticker_symbol", "classification",
            "probability_target_before_stop", "confidence",
            "entry_price", "stop_loss", "target_price",
        ]].to_string(index=False))

    print(f"\nLEDGER: {args.output} | rows={len(frame)}")
    print("NOTE: these are classifications only. Final trade outcomes require the forward validation window and must not be inferred from the current session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
