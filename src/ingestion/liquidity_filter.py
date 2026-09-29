"""Deterministic liquidity gate for the NSE swing universe.

The NSE security master defines what is a listed equity security. This module
answers a different question: whether recent traded activity is sufficient for
a 5-15 trading-day swing strategy.

No symbol list is hard-coded here. The gate is calculated from observed daily
OHLCV data and all thresholds are configurable.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class LiquidityConfig:
    lookback_days: int = 60
    min_observations: int = 40
    min_price: float = 20.0
    min_average_daily_turnover: float = 10_000_000.0  # INR 1 crore
    min_median_daily_turnover: float = 5_000_000.0   # INR 50 lakh
    max_zero_volume_fraction: float = 0.10


def evaluate_liquidity(
    candles: pd.DataFrame,
    config: LiquidityConfig = LiquidityConfig(),
) -> dict:
    """Evaluate recent liquidity without looking into future observations.

    Required columns: close_price and volume_traded. If a date/timestamp column
    exists, the latest lookback window is selected from it.
    """
    required = {"close_price", "volume_traded"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Liquidity frame missing columns: {sorted(missing)}")

    df = candles.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    elif "date" in df.columns:
        df = df.sort_values("date")

    df = df.tail(config.lookback_days).copy()
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["volume_traded"] = pd.to_numeric(df["volume_traded"], errors="coerce")
    df = df.dropna(subset=["close_price", "volume_traded"])

    if len(df) < config.min_observations:
        return {
            "eligible": False,
            "reason": "INSUFFICIENT_LIQUIDITY_HISTORY",
            "observations": len(df),
        }

    turnover = df["close_price"] * df["volume_traded"]
    avg_turnover = float(turnover.mean())
    median_turnover = float(turnover.median())
    zero_volume_fraction = float((df["volume_traded"] <= 0).mean())
    latest_price = float(df["close_price"].iloc[-1])

    checks = {
        "minimum_price": latest_price >= config.min_price,
        "average_turnover": avg_turnover >= config.min_average_daily_turnover,
        "median_turnover": median_turnover >= config.min_median_daily_turnover,
        "zero_volume": zero_volume_fraction <= config.max_zero_volume_fraction,
    }
    failed = [name for name, passed in checks.items() if not passed]

    return {
        "eligible": not failed,
        "reason": "LIQUIDITY_OK" if not failed else "LIQUIDITY_FAILED",
        "failed_checks": failed,
        "observations": len(df),
        "latest_price": latest_price,
        "average_daily_turnover": avg_turnover,
        "median_daily_turnover": median_turnover,
        "zero_volume_fraction": zero_volume_fraction,
    }
