"""Shared signal-level eligibility gates used by backtests and prediction datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.ingestion.nse_delivery_client import NSEArchiveDeliveryClient
from src.regime.market_regime import MarketRegimeEngine
from src.regime.sector_strength import RelativeStrengthEngine


@dataclass(frozen=True)
class TradeEligibility:
    """Deterministic signal-level eligibility result.

    Portfolio-state gates such as available slots, sector capacity, cash,
    position sizing, and portfolio circuit breakers are intentionally excluded:
    those depend on the simulated/live portfolio state rather than the signal.
    """

    eligible: bool
    reason: str
    risk_reward_ratio: float
    delivery_pass: bool
    regime_pass: bool
    rs_pass: bool


def evaluate_signal_eligibility(
    *,
    signal: Any,
    candidate_data: Dict[str, Any],
    nifty_slice: pd.DataFrame,
    is_expiry_week: bool,
    min_reward_risk: float = 2.0,
    enable_regime_filter: bool = True,
    enable_rs_filter: bool = True,
) -> TradeEligibility:
    """Apply the deterministic signal gates used by the backtest evening scan."""
    if enable_regime_filter and not nifty_slice.empty:
        regime = MarketRegimeEngine.evaluate_nifty_regime(nifty_slice)
        if not regime.get("allow_swing_longs", True):
            return TradeEligibility(False, "MARKET_REGIME_BLOCK", 0.0, False, False, False)

    delivery_history = candidate_data.get("delivery_history", [])
    today_delivery = candidate_data.get("today_delivery")
    if not delivery_history or today_delivery is None:
        return TradeEligibility(False, "DELIVERY_DATA_MISSING", 0.0, False, True, False)

    del_check = NSEArchiveDeliveryClient().calculate_delivery_expansion(
        recent_delivery_history=delivery_history,
        today_delivery=today_delivery,
        is_expiry_week=is_expiry_week,
    )
    if not del_check["is_institutional_accumulation"]:
        return TradeEligibility(False, "DELIVERY_EXPANSION_BLOCK", 0.0, False, True, False)

    risk_per_share = float(signal.entry_price) - float(signal.stop_loss)
    if risk_per_share <= 0:
        return TradeEligibility(False, "INVALID_RISK_PER_SHARE", 0.0, True, True, False)

    rr_ratio = (float(signal.target_price) - float(signal.entry_price)) / risk_per_share
    if rr_ratio < min_reward_risk:
        return TradeEligibility(False, "REWARD_RISK_BLOCK", rr_ratio, True, True, False)

    if enable_rs_filter:
        sector_series = candidate_data.get("sector_series")
        nifty_series = candidate_data.get("nifty_series")
        df_stock = candidate_data.get("ohlcv")
        if (
            sector_series is not None
            and nifty_series is not None
            and len(sector_series) >= 50
            and len(nifty_series) >= 50
            and df_stock is not None
        ):
            stock_series = df_stock["close"] if "close" in df_stock.columns else df_stock["Close"]
            rs_result = RelativeStrengthEngine.evaluate_dual_layer_relative_strength(
                stock_close=stock_series,
                sector_close=sector_series,
                nifty_close=nifty_series,
                stock_symbol=candidate_data.get("ticker_symbol", "UNKNOWN"),
                sector_name=candidate_data.get("sector", "UNKNOWN"),
            )
            if not rs_result["passes_rs_filter"]:
                return TradeEligibility(False, "RELATIVE_STRENGTH_BLOCK", rr_ratio, True, True, False)

    return TradeEligibility(True, "ELIGIBLE", rr_ratio, True, True, True)
