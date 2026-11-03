from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class StrategyMetrics:
    trade_count: int
    coverage: float
    win_rate: float
    average_r: float
    median_r: float
    expected_r: float
    profit_factor: float
    cumulative_r: float
    max_drawdown_r: float
    average_holding_days: float
    target_hit_rate: float
    stop_hit_rate: float
    time_expiry_rate: float
    mean_mfe_r: float
    mean_mae_r: float

def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]

def profit_factor(r_multiples: Sequence[float]) -> float:
    arr = _clean(r_multiples)
    gross_profit = float(arr[arr > 0].sum()) if arr.size else 0.0
    gross_loss = float(-arr[arr < 0].sum()) if arr.size else 0.0
    return (gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit else 0.0)

def max_drawdown_r(r_multiples: Sequence[float]) -> float:
    arr = _clean(r_multiples)
    if not arr.size:
        return 0.0
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    return float((np.r_[0.0, equity] - peak).min())

def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    p, y = np.asarray(probabilities, float), np.asarray(outcomes, float)
    mask = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else float("nan")

def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    p, y = np.asarray(probabilities, float), np.asarray(outcomes, float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if not p.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        inside = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if inside.any():
            ece += (inside.sum() / p.size) * abs(float(p[inside].mean()) - float(y[inside].mean()))
    return float(ece)

def top_k_lift(probabilities: Sequence[float], outcomes: Sequence[int], fraction: float) -> tuple[float, float, float]:
    p, y = np.asarray(probabilities, float), np.asarray(outcomes, float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if not p.size or not 0.0 < fraction <= 1.0:
        return float("nan"), float("nan"), float("nan")
    k = max(1, int(np.ceil(p.size * fraction)))
    top = y[np.argsort(-p, kind="mergesort")[:k]]
    baseline, precision = float(y.mean()), float(top.mean())
    return precision, baseline, precision / baseline if baseline > 0 else float("inf")

def strategy_metrics(r_multiples: Sequence[float], holding_days: Sequence[float], mfe_r: Sequence[float], mae_r: Sequence[float], coverage: float, exit_reasons: Sequence[str]) -> StrategyMetrics:
    r, hold, mfe, mae = map(_clean, (r_multiples, holding_days, mfe_r, mae_r))
    reasons = np.asarray(list(exit_reasons), dtype=str)
    n = int(r.size)
    return StrategyMetrics(
        trade_count=n, coverage=float(coverage), win_rate=float(np.mean(r > 0)) if n else 0.0,
        average_r=float(r.mean()) if n else 0.0, median_r=float(np.median(r)) if n else 0.0,
        expected_r=float(r.mean()) if n else 0.0, profit_factor=profit_factor(r), cumulative_r=float(r.sum()) if n else 0.0,
        max_drawdown_r=max_drawdown_r(r), average_holding_days=float(hold.mean()) if hold.size else 0.0,
        target_hit_rate=float(np.mean(np.char.find(reasons, "TARGET_HIT") >= 0)) if reasons.size else 0.0,
        stop_hit_rate=float(np.mean(np.char.find(reasons, "STOP_LOSS") >= 0)) if reasons.size else 0.0,
        time_expiry_rate=float(np.mean(np.char.find(reasons, "TIME_EXIT") >= 0)) if reasons.size else 0.0,
        mean_mfe_r=float(mfe.mean()) if mfe.size else 0.0, mean_mae_r=float(mae.mean()) if mae.size else 0.0,
    )
