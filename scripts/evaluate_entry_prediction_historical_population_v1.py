"""Validate the candidate model on the same historical signal population used for training.

The live shadow replay intentionally observes only TREND_PULLBACK because the live
engine currently locks production setup selection to that setup. The training
population, however, is built from all six validated setup detectors. This harness
therefore evaluates the candidate model directly against the historical population
used by the dataset builder, while fetching real daily candles from FYERS.

This is diagnostic only. It does not change EntryGate, production thresholds, or
live order behaviour.
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.ingestion.market_data_client import MarketDataClient
from src.prediction.dataset import _candidate_signals
from src.prediction.engine import PredictionFeatureBuilder

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_MODEL = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")
DEFAULT_OUTPUT = Path("data/prediction/historical_population_shadow_v1.csv")
HORIZON_SESSIONS = 15
REPLAY_SESSIONS = 180
WARMUP_SESSIONS = 320
CALENDAR_LOOKBACK_DAYS = 1100


def _value(signal: Any, name: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(name, default)
    return getattr(signal, name, default)


def _label(features: pd.DataFrame, entry_idx: int, signal: Any) -> Optional[dict]:
    entry = float(_value(signal, "entry_price", np.nan))
    stop = float(_value(signal, "stop_loss", np.nan))
    target = float(_value(signal, "target_price", np.nan))
    risk = entry - stop
    if not np.isfinite(entry) or not np.isfinite(stop) or not np.isfinite(target):
        return None
    if risk <= 0 or target <= entry:
        return None

    future = features.iloc[entry_idx + 1 : entry_idx + 1 + HORIZON_SESSIONS]
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

        # Match entry_prediction_dataset_v1: adverse barrier first on same bar.
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

    return {
        "label_target_before_stop": label,
        "outcome_valid": int(not unresolved),
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "days_to_target": days_to_target or 0,
        "days_to_stop": days_to_stop or 0,
    }


def _load_model(path: Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("model_version") != "entry_prediction_extra_trees_candidate_v1":
        raise ValueError("Unsupported candidate model version")
    if bundle.get("production_ready") is not False or bundle.get("promotion_status") != "candidate_only":
        raise ValueError("Candidate model metadata is not safely candidate-only")
    if list(bundle.get("feature_columns", [])) != list(PredictionFeatureBuilder.FEATURE_COLUMNS):
        raise ValueError("Candidate feature schema does not match PredictionFeatureBuilder")
    return bundle


def evaluate_symbol(symbol: str, sessions: int, bundle: dict) -> list[dict]:
    end_date = datetime.now(IST).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=CALENDAR_LOOKBACK_DAYS)

    with MarketDataClient() as client:
        if not client.authenticate():
            raise RuntimeError("FYERS authentication is unavailable")
        raw = client.fetch_historical_candles(symbol, start_date, end_date, resolution="D")

    if len(raw) < WARMUP_SESSIONS + sessions + HORIZON_SESSIONS:
        raise RuntimeError(f"Not enough history for {symbol}: {len(raw)} sessions returned")

    raw = raw.reset_index(drop=True)
    features = TechnicalAnalysisEngine.compute_all_features(raw.copy())
    features["timestamp"] = raw["timestamp"].values

    replay_start = len(features) - sessions - HORIZON_SESSIONS
    replay_end = len(features) - HORIZON_SESSIONS
    start_idx = max(WARMUP_SESSIONS, replay_start)
    rows: list[dict] = []

    for i in range(start_idx, replay_end):
        window = features.iloc[: i + 1]
        signals = _candidate_signals(symbol, window)
        for signal in signals:
            outcome = _label(features, i, signal)
            if outcome is None:
                continue
            feature_row = PredictionFeatureBuilder.build(signal, window).reindex(
                columns=bundle["feature_columns"], fill_value=0.0
            )
            probability = float(bundle["model"].predict_proba(feature_row)[0, 1])
            timestamp = pd.Timestamp(features.iloc[i]["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(IST)
            rows.append(
                {
                    "ticker_symbol": symbol,
                    "trade_date": timestamp.tz_convert(IST).date(),
                    "setup_type": str(_value(signal, "setup_type", "")),
                    "entry_price": float(_value(signal, "entry_price", np.nan)),
                    "stop_loss": float(_value(signal, "stop_loss", np.nan)),
                    "target_price": float(_value(signal, "target_price", np.nan)),
                    "probability_target_before_stop": probability,
                    "confidence": abs(probability - 0.5) * 2.0,
                    **outcome,
                }
            )

    print(f"POPULATION REPLAY | {symbol} | candidates={len(rows)} | replay_sessions={sessions}")
    return rows


def _lift(frame: pd.DataFrame, fraction: float) -> Optional[float]:
    if frame.empty:
        return None
    baseline = float(frame["label_target_before_stop"].mean())
    if baseline <= 0:
        return None
    n = max(1, int(np.ceil(len(frame) * fraction)))
    top = frame.sort_values("probability_target_before_stop", ascending=False).head(n)
    return float(top["label_target_before_stop"].mean() / baseline)


def summarize(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("DECISION: zero evaluable historical candidates; investigate setup population before model work.")
        return

    y = frame["label_target_before_stop"].astype(int)
    p = frame["probability_target_before_stop"].astype(float)
    print("\n=== HISTORICAL POPULATION SHADOW EVALUATION ===")
    print(f"rows={len(frame)}")
    print(f"tickers={frame['ticker_symbol'].nunique()}")
    print(f"positive_rate={y.mean():.6f}")
    print(f"mean_probability={p.mean():.6f}")
    if y.nunique() == 2:
        print(f"auc={roc_auc_score(y, p):.6f}")
        print(f"ap={average_precision_score(y, p):.6f}")
    else:
        print("auc=NA")
        print("ap=NA")
    print(f"brier={brier_score_loss(y, p):.6f}")
    for pct in (0.01, 0.03, 0.05, 0.10):
        lift = _lift(frame, pct)
        label = f"top_{int(pct * 100)}pct_lift"
        print(f"{label}={lift:.6f}" if lift is not None else f"{label}=NA")

    print("\nSETUP BREAKDOWN")
    print(
        frame.groupby("setup_type")
        .agg(
            predictions=("label_target_before_stop", "size"),
            actual_rate=("label_target_before_stop", "mean"),
            mean_probability=("probability_target_before_stop", "mean"),
        )
        .sort_values("predictions", ascending=False)
        .to_string()
    )
    print("\nDECISION: diagnostic only; candidate remains production_ready=False and candidate_only.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--sessions", type=int, default=REPLAY_SESSIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.sessions < 1:
        raise ValueError("--sessions must be >= 1")

    bundle = _load_model(args.model)
    print(f"=== GANESHA V1 HISTORICAL POPULATION | MODEL AVAILABLE: True ===")

    rows: list[dict] = []
    for symbol in args.symbols:
        rows.extend(evaluate_symbol(symbol, args.sessions, bundle))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"HISTORICAL POPULATION LEDGER: {args.output} | rows={len(frame)}")

    summarize(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
