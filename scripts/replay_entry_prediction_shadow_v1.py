"""Replay real FYERS daily candles through the live analysis + shadow model.

This is a read-only diagnostic harness. It uses real historical FYERS data,
replays it chronologically through the same LiveAnalysisEngine used by the
live monitor, and runs the candidate model only when the validated live setup
scanner produces a TREND_PULLBACK signal.

It never places orders, changes EntryGate, or promotes the candidate model.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.market_data_client import MarketDataClient
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.prediction.candidate_shadow import CandidateShadowPredictor

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
DEFAULT_SESSIONS = 60
WARMUP_SESSIONS = 320
CALENDAR_LOOKBACK_DAYS = 900


def _as_replay_candle(symbol: str, row: pd.Series) -> OneMinuteCandle:
    """Convert one real daily FYERS bar into a completed-session candle."""
    session_date = pd.Timestamp(row["timestamp"]).tz_convert(IST).date()
    return OneMinuteCandle(
        symbol=symbol,
        timestamp=datetime.combine(session_date, time(15, 29), tzinfo=IST),
        open_price=float(row["open_price"]),
        high_price=float(row["high_price"]),
        low_price=float(row["low_price"]),
        close_price=float(row["close_price"]),
        volume_traded=int(row["volume_traded"]),
        tick_count=1,
    )


def replay_symbol(symbol: str, sessions: int, predictor: CandidateShadowPredictor) -> int:
    """Replay one symbol and return the number of shadow predictions emitted."""
    end_date = datetime.now(IST).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=CALENDAR_LOOKBACK_DAYS)

    with MarketDataClient() as client:
        if not client.authenticate():
            raise RuntimeError("FYERS authentication is unavailable.")
        history = client.fetch_historical_candles(symbol, start_date, end_date, resolution="D")

    if history.empty:
        raise RuntimeError(f"No historical candles returned for {symbol}")

    required = WARMUP_SESSIONS + sessions
    if len(history) < required:
        raise RuntimeError(
            f"Not enough history for {symbol}: got {len(history)} sessions, need at least {required}."
        )

    history = history.tail(required).reset_index(drop=True)
    replay_start = len(history) - sessions
    state = {"index": -1, "predictions": 0, "signals": 0}

    def on_signals(signal_symbol: str, signals: list) -> None:
        if state["index"] < replay_start:
            return
        state["signals"] += len(signals)
        features = analysis.dataframe(signal_symbol)
        for signal in signals:
            shadow = predictor.predict(signal, features)
            if not shadow.available:
                continue
            state["predictions"] += 1
            print(
                "SHADOW REPLAY PREDICTION | "
                f"{shadow.symbol} | {shadow.setup_type} | "
                f"P={shadow.probability_target_before_stop:.4f} | "
                f"CONF={shadow.confidence:.4f} | "
                f"MODEL={shadow.model_version}",
                flush=True,
            )

    analysis = LiveAnalysisEngine(signal_handler=on_signals)

    print(
        f"REPLAY RANGE | {symbol} | "
        f"warmup={history.iloc[0]['timestamp'].date()} | "
        f"replay={history.iloc[replay_start]['timestamp'].date()} -> "
        f"{history.iloc[-1]['timestamp'].date()} | sessions={sessions}",
        flush=True,
    )

    for index, row in history.iterrows():
        state["index"] = index
        analysis.on_candle(_as_replay_candle(symbol, row))

    print(
        f"REPLAY COMPLETE | {symbol} | signals={state['signals']} | "
        f"shadow_predictions={state['predictions']}",
        flush=True,
    )
    return int(state["predictions"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--shadow-model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    if args.sessions < 1:
        raise ValueError("--sessions must be >= 1")

    predictor = CandidateShadowPredictor(args.shadow_model)
    print(f"=== GANESHA V1 SHADOW REPLAY | MODEL AVAILABLE: {predictor.available} ===")

    total = 0
    for symbol in args.symbols:
        total += replay_symbol(symbol, args.sessions, predictor)

    print(f"SHADOW REPLAY TOTAL PREDICTIONS: {total}")
    print("DECISION: diagnostic only; EntryGate and production model remain unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
