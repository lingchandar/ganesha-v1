"""Diagnostic validation for entry prediction v1.

Consumes the leakage-safe OOS prediction file produced by the strategy
validator. This is diagnostic only: it does not change live/backtest behavior
or introduce a production threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _score_column(frame: pd.DataFrame) -> str:
    """Return the evaluator's probability column without renaming data blindly."""
    if "prediction" in frame.columns:
        return "prediction"
    if "probability" in frame.columns:
        return "probability"
    raise ValueError("Missing prediction score column: expected 'probability' or 'prediction'")


def _metric(frame: pd.DataFrame, score_col: str | None = None) -> dict[str, float]:
    score_col = score_col or _score_column(frame)
    frame = frame.dropna(subset=[score_col, "label_target_before_stop"])
    if frame.empty or frame["label_target_before_stop"].nunique() < 2:
        return {"rows": float(len(frame)), "auc": np.nan, "ap": np.nan}
    return {
        "rows": float(len(frame)),
        "auc": float(roc_auc_score(frame["label_target_before_stop"], frame[score_col])),
        "ap": float(average_precision_score(frame["label_target_before_stop"], frame[score_col])),
    }


def _daily_top_k(frame: pd.DataFrame, k: int, score_col: str | None = None) -> tuple[float, float, int]:
    score_col = score_col or _score_column(frame)
    hits: list[float] = []
    for _, day in frame.groupby("trade_date", sort=True):
        ranked = day.sort_values(score_col, ascending=False).head(k)
        if len(ranked):
            hits.append(float(ranked["label_target_before_stop"].mean()))
    if not hits:
        return np.nan, np.nan, 0
    base = float(frame["label_target_before_stop"].mean())
    rate = float(np.mean(hits))
    return rate, (rate / base if base else np.nan), len(hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    score_col = _score_column(df)
    required = {
        "trade_date", "ticker_symbol", "setup_type", "label_target_before_stop", "fold",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["trade_date", "ticker_symbol", "setup_type"]).reset_index(drop=True)

    print("ENTRY_PREDICTION_V1_DIAGNOSTICS")
    print(f"score_column={score_col}")
    print(f"rows={len(df)} dates={df.trade_date.nunique()} tickers={df.ticker_symbol.nunique()}")
    print(f"pooled={_metric(df, score_col)}")

    print("\nFOLD_BY_FOLD")
    for fold, part in df.groupby("fold", sort=True):
        print(f"fold={fold} {_metric(part, score_col)}")

    print("\nSETUP_BY_SETUP")
    for setup, part in df.groupby("setup_type", sort=True):
        metric = _metric(part, score_col)
        print(f"setup={setup} {metric}")

    print("\nDEDUPLICATION_EFFECT")
    raw = _metric(df, score_col)
    best = (
        df.sort_values(["trade_date", "ticker_symbol", score_col], ascending=[True, True, False])
        .drop_duplicates(["trade_date", "ticker_symbol"], keep="first")
        .copy()
    )
    print(f"all_setups={raw}")
    print(f"one_best_setup_per_ticker_day={_metric(best, score_col)} rows={len(best)}")

    print("\nDAILY_TOP_K_ON_RAW_OOS")
    for k in (1, 3, 5, 10):
        rate, lift, days = _daily_top_k(df, k, score_col)
        print(f"top_{k}=positive_rate={rate:.4f} lift={lift:.2f}x days={days}")

    print("\nBASELINE_RANKINGS")
    baseline_cols = [
        c for c in (
            "risk_reward_ratio", "volume_surge_multiple", "return_5d",
            "return_20d", "price_vs_ema20_pct", "price_vs_ema50_pct",
        ) if c in df.columns
    ]
    for col in baseline_cols:
        work = df[["label_target_before_stop", col]].dropna().copy()
        if work[col].nunique() < 2:
            continue
        auc = roc_auc_score(work["label_target_before_stop"], work[col])
        inverse = roc_auc_score(work["label_target_before_stop"], -work[col])
        print(f"feature={col} auc={auc:.4f} inverse_auc={inverse:.4f}")

    print("\nRANKING_DIRECTION_CHECK")
    auc = _metric(df, score_col)["auc"]
    print(f"model_auc={auc:.4f} inverted_auc={1.0 - auc:.4f}" if not np.isnan(auc) else "model_auc=nan")
    print("No production decision is made by this diagnostic.")


if __name__ == "__main__":
    main()
