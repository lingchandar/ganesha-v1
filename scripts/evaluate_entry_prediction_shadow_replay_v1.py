"""Evaluate candidate shadow predictions against real forward outcomes.

This is a read-only validation harness. It replays real FYERS daily candles through
LiveAnalysisEngine, records every candidate TREND_PULLBACK prediction, resolves
that prediction with the same 15-session target-before-stop policy used by the
entry-prediction dataset, and reports diagnostic metrics.

It never changes EntryGate, production thresholds, or live order behavior.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.market_data_client import MarketDataClient
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.prediction.candidate_shadow import CandidateShadowPredictor

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
DEFAULT_OUTPUT = Path("data/prediction/shadow_replay_v1.csv")
HORIZON_SESSIONS = 15
WARMUP_SESSIONS = 320
CALENDAR_LOOKBACK_DAYS = 900


def _as_replay_candle(symbol: str, row: pd.Series) -> OneMinuteCandle:
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


def _signal_value(signal: Any, name: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(name, default)
    return getattr(signal, name, default)


def _resolve_outcome(history: pd.DataFrame, signal_date: date, signal: Any) -> Optional[dict]:
    entry = float(_signal_value(signal, "entry_price", np.nan))
    stop = float(_signal_value(signal, "stop_loss", np.nan))
    target = float(_signal_value(signal, "target_price", np.nan))
    risk = entry - stop
    if not np.isfinite(entry) or not np.isfinite(stop) or not np.isfinite(target):
        return None
    if risk <= 0 or target <= entry:
        return None

    dates = pd.to_datetime(history["timestamp"]).dt.tz_convert(IST).dt.date
    future = history.loc[dates > signal_date].head(HORIZON_SESSIONS)
    if future.empty:
        return None

    target_hit = False
    stop_hit = False
    days_to_target = None
    days_to_stop = None
    mfe_r = 0.0
    mae_r = 0.0

    for offset, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high_price"])
        low = float(row["low_price"])
        mfe_r = max(mfe_r, (high - entry) / risk)
        mae_r = min(mae_r, (low - entry) / risk)

        # Match entry_prediction_dataset_v1: adverse barrier is checked first.
        if not stop_hit and low <= stop:
            stop_hit = True
            days_to_stop = offset
        if not target_hit and high >= target:
            target_hit = True
            days_to_target = offset

        if stop_hit or target_hit:
            break

    unresolved = not target_hit and not stop_hit
    if stop_hit and target_hit:
        label = int(days_to_target < days_to_stop)
    elif target_hit:
        label = 1
    else:
        label = 0

    horizon_end_close = None
    horizon_end_r = None
    if unresolved and len(future) == HORIZON_SESSIONS:
        horizon_end_close = float(future.iloc[-1]["close_price"])
        horizon_end_r = (horizon_end_close - entry) / risk

    return {
        "label_target_before_stop": label,
        "outcome_valid": int(not unresolved),
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "days_to_target": days_to_target or 0,
        "days_to_stop": days_to_stop or 0,
        "horizon_end_close": horizon_end_close,
        "horizon_end_r": horizon_end_r,
    }


def replay_symbol(symbol: str, sessions: int, predictor: CandidateShadowPredictor) -> list[dict]:
    end_date = datetime.now(IST).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=CALENDAR_LOOKBACK_DAYS)

    with MarketDataClient() as client:
        if not client.authenticate():
            raise RuntimeError("FYERS authentication is unavailable.")
        history = client.fetch_historical_candles(symbol, start_date, end_date, resolution="D")

    if history.empty:
        raise RuntimeError(f"No historical candles returned for {symbol}")

    required = WARMUP_SESSIONS + sessions + HORIZON_SESSIONS
    if len(history) < required:
        raise RuntimeError(
            f"Not enough history for {symbol}: got {len(history)} sessions, need at least {required}."
        )

    history = history.reset_index(drop=True)
    replay_start = len(history) - sessions - HORIZON_SESSIONS
    replay_end = len(history) - HORIZON_SESSIONS
    replay_history = history.iloc[:replay_end].tail(WARMUP_SESSIONS + sessions).reset_index(drop=True)
    replay_start = len(replay_history) - sessions
    state = {"index": -1}
    records: list[dict] = []

    def on_signals(signal_symbol: str, signals: list) -> None:
        if state["index"] < replay_start:
            return
        features = analysis.dataframe(signal_symbol)
        if features.empty:
            return
        latest_timestamp = pd.Timestamp(features.iloc[-1]["timestamp"])
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.tz_localize(IST)
        signal_date = latest_timestamp.tz_convert(IST).date()

        for signal in signals:
            shadow = predictor.predict(signal, features)
            if not shadow.available or shadow.probability_target_before_stop is None:
                continue
            outcome = _resolve_outcome(history, signal_date, signal)
            if outcome is None:
                continue
            records.append(
                {
                    "ticker_symbol": shadow.symbol,
                    "trade_date": signal_date,
                    "setup_type": shadow.setup_type,
                    "entry_price": float(_signal_value(signal, "entry_price", np.nan)),
                    "stop_loss": float(_signal_value(signal, "stop_loss", np.nan)),
                    "target_price": float(_signal_value(signal, "target_price", np.nan)),
                    "probability_target_before_stop": float(shadow.probability_target_before_stop),
                    "confidence": float(shadow.confidence or 0.0),
                    "model_version": shadow.model_version,
                    **outcome,
                }
            )

    analysis = LiveAnalysisEngine(signal_handler=on_signals)

    print(
        f"REPLAY RANGE | {symbol} | "
        f"warmup={replay_history.iloc[0]['timestamp'].date()} | "
        f"replay={replay_history.iloc[replay_start]['timestamp'].date()} -> "
        f"{replay_history.iloc[-1]['timestamp'].date()} | sessions={sessions}",
        flush=True,
    )

    for index, row in replay_history.iterrows():
        state["index"] = index
        analysis.on_candle(_as_replay_candle(symbol, row))

    print(f"REPLAY COMPLETE | {symbol} | evaluated_predictions={len(records)}", flush=True)
    return records


def _lift(frame: pd.DataFrame, fraction: float) -> Optional[float]:
    if frame.empty:
        return None
    n = max(1, int(np.ceil(len(frame) * fraction)))
    top = frame.sort_values("probability_target_before_stop", ascending=False).head(n)
    baseline = float(frame["label_target_before_stop"].mean())
    if baseline <= 0:
        return None
    return float(top["label_target_before_stop"].mean() / baseline)


def summarize(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("DECISION: no evaluable shadow predictions were produced.")
        return

    y = frame["label_target_before_stop"].astype(int)
    p = frame["probability_target_before_stop"].astype(float)
    print("\n=== ENTRY PREDICTION SHADOW EVALUATION ===")
    print(f"rows={len(frame)}")
    print(f"positive_rate={y.mean():.6f}")
    print(f"mean_probability={p.mean():.6f}")
    if y.nunique() == 2:
        print(f"auc={roc_auc_score(y, p):.6f}")
        print(f"ap={average_precision_score(y, p):.6f}")
    else:
        print("auc=NA (only one outcome class)")
        print("ap=NA (only one outcome class)")
    print(f"brier={brier_score_loss(y, p):.6f}")
    for pct in (0.01, 0.03, 0.05, 0.10):
        value = _lift(frame, pct)
        print(f"top_{int(pct * 100)}pct_lift={value:.6f}" if value is not None else f"top_{int(pct * 100)}pct_lift=NA")

    frame["probability_bucket"] = pd.cut(
        p,
        bins=[-np.inf, 0.10, 0.15, 0.20, 0.30, np.inf],
        labels=["<=0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", ">0.30"],
        right=True,
    )
    buckets = (
        frame.groupby("probability_bucket", observed=False)
        .agg(
            predictions=("label_target_before_stop", "size"),
            actual_rate=("label_target_before_stop", "mean"),
            mean_probability=("probability_target_before_stop", "mean"),
        )
        .reset_index()
    )
    print("\nPROBABILITY BUCKETS")
    print(buckets.to_string(index=False))

    print("\nOUTCOME VALIDITY")
    print(frame.groupby("outcome_valid").size().rename("rows").to_string())
    print("DECISION: diagnostic only; candidate remains production_ready=False and candidate_only.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--sessions", type=int, default=180)
    parser.add_argument("--shadow-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.sessions < 1:
        raise ValueError("--sessions must be >= 1")

    predictor = CandidateShadowPredictor(args.shadow_model)
    print(f"=== GANESHA V1 SHADOW EVALUATION | MODEL AVAILABLE: {predictor.available} ===")

    rows: list[dict] = []
    for symbol in args.symbols:
        rows.extend(replay_symbol(symbol, args.sessions, predictor))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"SHADOW LEDGER: {args.output} | rows={len(frame)}")

    summarize(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
