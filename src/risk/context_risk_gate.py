"""
GANESHA V1 — Context & Shock Risk Gate.

The technical setup is not the whole market. This gate detects recent abnormal
price/volatility shocks and prevents a normal swing setup from being treated as
normal immediately after an exceptional move.

This is intentionally conservative and point-in-time safe:
- only candles up to the scan date are inspected;
- no news or future information is inferred;
- the gate can be disabled to reproduce the original V1 baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextRiskDecision:
    """Point-in-time decision for a stock setup."""

    allowed: bool
    risk_level: str
    reason: str
    metrics: Dict[str, float]


class ContextRiskGate:
    """
    Detects abnormal stock-level conditions that can make a normal swing setup
    unreliable, such as a recent crash/gap or volatility shock.

    This is not a news detector. A news/geopolitical feed can be added later;
    until then, market-price anomalies provide a deterministic safety layer.
    """

    MIN_BARS = 30
    RETURN_SHOCK_PCT = 5.0
    ATR_SHOCK_MULTIPLE = 2.5
    GAP_SHOCK_PCT = 4.0
    VOLATILITY_RATIO = 2.0
    COOLDOWN_SESSIONS = 2

    @classmethod
    def evaluate(cls, df: pd.DataFrame) -> ContextRiskDecision:
        """Evaluate only the information available through the latest candle."""
        if len(df) < cls.MIN_BARS:
            return ContextRiskDecision(
                allowed=False,
                risk_level="UNKNOWN",
                reason="INSUFFICIENT_CONTEXT_HISTORY",
                metrics={},
            )

        frame = df.copy()
        close_col = "close" if "close" in frame.columns else "close_price"
        open_col = "open" if "open" in frame.columns else "open_price"
        high_col = "high" if "high" in frame.columns else "high_price"
        low_col = "low" if "low" in frame.columns else "low_price"

        close = pd.to_numeric(frame[close_col], errors="coerce")
        open_ = pd.to_numeric(frame[open_col], errors="coerce")
        high = pd.to_numeric(frame[high_col], errors="coerce")
        low = pd.to_numeric(frame[low_col], errors="coerce")

        prev_close = close.shift(1)
        daily_return_pct = close.pct_change() * 100.0
        gap_pct = (open_ / prev_close - 1.0).abs() * 100.0

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range.rolling(14).mean()
        atr_pct = atr_14 / close * 100.0
        baseline_atr_pct = atr_pct.rolling(20).median()
        volatility_ratio = atr_pct / baseline_atr_pct.replace(0, np.nan)

        latest_return = float(daily_return_pct.iloc[-1])
        latest_gap = float(gap_pct.iloc[-1])
        latest_atr_pct = float(atr_pct.iloc[-1])
        latest_vol_ratio = float(volatility_ratio.iloc[-1])

        recent = pd.DataFrame(
            {
                "return": daily_return_pct,
                "gap": gap_pct,
                "atr_pct": atr_pct,
                "vol_ratio": volatility_ratio,
            }
        ).tail(cls.COOLDOWN_SESSIONS)

        recent_return_shock = bool((recent["return"].abs() >= cls.RETURN_SHOCK_PCT).any())
        recent_gap_shock = bool((recent["gap"] >= cls.GAP_SHOCK_PCT).any())
        recent_atr_shock = bool((recent["vol_ratio"] >= cls.VOLATILITY_RATIO).any())

        if recent_return_shock or recent_gap_shock or recent_atr_shock:
            reasons = []
            if recent_return_shock:
                reasons.append("RECENT_RETURN_SHOCK")
            if recent_gap_shock:
                reasons.append("RECENT_GAP_SHOCK")
            if recent_atr_shock:
                reasons.append("VOLATILITY_EXPANSION")
            return ContextRiskDecision(
                allowed=False,
                risk_level="HIGH",
                reason=";".join(reasons),
                metrics={
                    "latest_return_pct": round(latest_return, 3),
                    "latest_gap_pct": round(latest_gap, 3),
                    "latest_atr_pct": round(latest_atr_pct, 3),
                    "latest_volatility_ratio": round(latest_vol_ratio, 3),
                },
            )

        return ContextRiskDecision(
            allowed=True,
            risk_level="NORMAL",
            reason="NORMAL_CONTEXT",
            metrics={
                "latest_return_pct": round(latest_return, 3),
                "latest_gap_pct": round(latest_gap, 3),
                "latest_atr_pct": round(latest_atr_pct, 3),
                "latest_volatility_ratio": round(latest_vol_ratio, 3),
            },
        )
