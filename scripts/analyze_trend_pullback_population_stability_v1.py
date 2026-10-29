"""Analyze Trend Pullback candidate-model stability and probability calibration.

Diagnostic only. Reads the historical population ledger produced by the 50-stock
Trend Pullback validation. It does not retrain, promote, change EntryGate, or
place orders.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_LEDGER = Path("data/prediction/historical_population_trend_pullback_shadow_v1.csv")
DEFAULT_BINS = 10
DEFAULT_BOOTSTRAPS = 1000
RANDOM_SEED = 42


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"rows": 0}
    y = frame["label_target_before_stop"].astype(int).to_numpy()
    p = frame["probability_target_before_stop"].astype(float).to_numpy()
    result = {"rows": float(len(frame)), "positive_rate": float(y.mean()), "mean_probability": float(p.mean()), "brier": float(brier_score_loss(y, p))}
    if np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["ap"] = float(average_precision_score(y, p))
    else:
        result["auc"] = float("nan")
        result["ap"] = float("nan")
    baseline = float(y.mean())
    for pct in (0.01, 0.03, 0.05, 0.10):
        key = f"top_{int(pct * 100)}pct_lift"
        if baseline <= 0:
            result[key] = float("nan")
            continue
        n = max(1, int(np.ceil(len(frame) * pct)))
        top = frame.sort_values("probability_target_before_stop", ascending=False).head(n)
        result[key] = float(top["label_target_before_stop"].mean() / baseline)
    return result


def _calibration(frame: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, dict[str, float]]:
    y = frame["label_target_before_stop"].astype(int)
    p = frame["probability_target_before_stop"].astype(float).clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = pd.cut(p, bins=edges, include_lowest=True, labels=False)
    rows = []
    ece = 0.0
    mce = 0.0
    for idx in range(bins):
        mask = bucket == idx
        count = int(mask.sum())
        if count == 0:
            continue
        mean_p = float(p[mask].mean())
        observed = float(y[mask].mean())
        gap = abs(mean_p - observed)
        ece += (count / len(frame)) * gap
        mce = max(mce, gap)
        rows.append({"bin": idx + 1, "range_low": float(edges[idx]), "range_high": float(edges[idx + 1]), "count": count, "mean_probability": mean_p, "actual_rate": observed, "absolute_gap": gap})
    return pd.DataFrame(rows), {"ece": ece, "mce": mce}


def _chronological_calibrator(frame: pd.DataFrame, bins: int) -> dict[str, float]:
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    split = int(len(ordered) * 2 / 3)
    train, test = ordered.iloc[:split], ordered.iloc[split:]
    if len(train) < 20 or len(test) < 20 or train["label_target_before_stop"].nunique() < 2 or test["label_target_before_stop"].nunique() < 2:
        return {"calibration_holdout_rows": float(len(test))}

    def logit(values: pd.Series) -> np.ndarray:
        x = values.astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
        return np.log(x / (1.0 - x)).reshape(-1, 1)

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logit(train["probability_target_before_stop"]), train["label_target_before_stop"])
    calibrated = model.predict_proba(logit(test["probability_target_before_stop"]))[:, 1]
    y_test = test["label_target_before_stop"].astype(int).to_numpy()
    calibrated_frame = test.copy()
    # Do not rename into a duplicate column: pandas then returns a 2-D object
    # for frame["probability_target_before_stop"], breaking pd.cut().
    calibrated_frame["probability_target_before_stop"] = calibrated
    _, cal_stats = _calibration(calibrated_frame, bins)
    return {"calibration_holdout_rows": float(len(test)), "calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0][0]), "calibration_holdout_brier": float(brier_score_loss(y_test, calibrated)), "calibration_holdout_ece": float(cal_stats["ece"])}


def _bootstrap(frame: pd.DataFrame, iterations: int, seed: int) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    n = len(frame)
    auc_values, lift_values = [], []
    for _ in range(iterations):
        sample = frame.iloc[rng.integers(0, n, n)]
        m = _metrics(sample)
        if np.isfinite(m["auc"]): auc_values.append(m["auc"])
        if np.isfinite(m["top_10pct_lift"]): lift_values.append(m["top_10pct_lift"])

    def interval(values: list[float]) -> tuple[float, float, float]:
        if not values: return (float("nan"), float("nan"), float("nan"))
        arr = np.asarray(values)
        return (float(np.median(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))
    return {"auc": interval(auc_values), "top_10pct_lift": interval(lift_values)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    args = parser.parse_args(argv)
    if args.bins < 2: raise ValueError("--bins must be >= 2")
    if args.bootstraps < 100: raise ValueError("--bootstraps must be >= 100")
    if not args.ledger.exists(): raise FileNotFoundError(f"Ledger not found: {args.ledger}")

    frame = pd.read_csv(args.ledger)
    required = {"ticker_symbol", "trade_date", "setup_type", "probability_target_before_stop", "label_target_before_stop"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"Ledger missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])

    overall = _metrics(frame)
    _, cal = _calibration(frame, args.bins)
    holdout_cal = _chronological_calibrator(frame, args.bins)
    boot = _bootstrap(frame, args.bootstraps, RANDOM_SEED)

    print("=== GANESHA V1 TREND PULLBACK STABILITY + CALIBRATION ===")
    print(f"ledger={args.ledger}")
    print(f"rows={int(overall['rows'])}")
    print(f"tickers={frame['ticker_symbol'].nunique()}")
    print(f"date_range={frame['trade_date'].min().date()} -> {frame['trade_date'].max().date()}")
    for key in ("positive_rate", "mean_probability", "auc", "ap", "brier", "top_1pct_lift", "top_3pct_lift", "top_5pct_lift", "top_10pct_lift"):
        print(f"{key}={overall[key]:.6f}")
    print(f"ece={cal['ece']:.6f}")
    print(f"mce={cal['mce']:.6f}")

    print("\n=== CHRONOLOGICAL STABILITY ===")
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    for idx, part in enumerate(np.array_split(ordered, 3), start=1):
        m = _metrics(part)
        print(f"third_{idx} | rows={int(m['rows'])} | positive_rate={m['positive_rate']:.6f} | mean_probability={m['mean_probability']:.6f} | auc={m['auc']:.6f} | top10_lift={m['top_10pct_lift']:.6f} | brier={m['brier']:.6f}")

    print("\n=== TICKER STABILITY ===")
    ticker_rows = []
    for ticker, group in frame.groupby("ticker_symbol"):
        ticker_rows.append({"ticker": ticker, **_metrics(group)})
    ticker_frame = pd.DataFrame(ticker_rows).sort_values("rows", ascending=False)
    print(ticker_frame[["ticker", "rows", "positive_rate", "auc", "top_10pct_lift", "brier"]].to_string(index=False))
    aucs = ticker_frame["auc"].replace([np.inf, -np.inf], np.nan).dropna()
    lifts = ticker_frame["top_10pct_lift"].replace([np.inf, -np.inf], np.nan).dropna()
    if not aucs.empty: print(f"ticker_auc_median={aucs.median():.6f} | min={aucs.min():.6f} | max={aucs.max():.6f} | positive_tickers={(aucs >= 0.5).sum()}/{len(aucs)}")
    if not lifts.empty: print(f"ticker_top10_lift_median={lifts.median():.6f} | min={lifts.min():.6f} | max={lifts.max():.6f}")

    print("\n=== PROBABILITY CALIBRATION HOLDOUT ===")
    for key, value in holdout_cal.items(): print(f"{key}={value:.6f}")
    print("Interpretation: calibration_slope near 1 and intercept near 0 are desirable; these are diagnostics, not promotion gates.")

    print("\n=== BOOTSTRAP 95% INTERVALS ===")
    for metric, (median, low, high) in boot.items(): print(f"{metric} | median={median:.6f} | 95% CI={low:.6f}..{high:.6f}")

    print("\nDECISION: diagnostic only; candidate remains production_ready=False and candidate_only.")
    print("NEXT: use stability/calibration evidence to decide whether a new training/OOS experiment is justified; do not promote from this report alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
