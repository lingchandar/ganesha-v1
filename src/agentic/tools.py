"""
GANESHA V1 — Deterministic Tool Calling Registry (Module 10, 14, 27)

DETERMINISTIC SEPARATION OF POWERS:
These functions provide the factual, mathematical ground-truth data for the
Gemini Agentic AI Orchestrator. The AI never calculates prices or indicators
on its own — it relies exclusively on these validated Python tools.
"""
from datetime import date
from typing import Any, Dict, List, Optional
import pandas as pd
from loguru import logger

from src.ingestion.catb_filter import CorporateActionTimingBuffer
from src.ingestion.expiry_calendar import NSEDerivativeExpiryFilter
from src.ingestion.nse_delivery_client import NSEArchiveDeliveryClient
from src.indicators.technical_indicators import TechnicalAnalysisEngine
from src.regime.market_regime import MarketRegimeEngine
from src.regime.sector_strength import RelativeStrengthEngine
from src.setups.swing_setups import SwingSetupScanner
from src.risk.exit_engine import PrecisionExitEngine


def get_market_regime(nifty_candles_df: pd.DataFrame, india_vix: Optional[float] = None) -> Dict[str, Any]:
    """
    Evaluates broader market regime from Nifty 50 historical daily candles.
    Returns regime classification (BULLISH, NEUTRAL, HIGH_VOLATILITY, BEARISH)
    and trade permission flag.
    """
    return MarketRegimeEngine.evaluate_nifty_regime(nifty_candles_df, india_vix)


def check_derivative_expiry(trade_date_iso: str) -> Dict[str, Any]:
    """
    Checks if trade_date falls in the final 3 sessions of monthly derivative expiry
    and returns threshold tightening parameters (1.725x delivery, +15% breakout clearance).
    """
    t_date = date.fromisoformat(trade_date_iso)
    expiry_filter = NSEDerivativeExpiryFilter()
    return expiry_filter.is_expiry_week_active(t_date)


def check_catb_blackout(ticker_symbol: str, trade_date_iso: str) -> Dict[str, Any]:
    """
    Checks if ticker_symbol is within the 5-day corporate action blackout window
    [Ex - 3 days, Ex + 2 days].
    """
    t_date = date.fromisoformat(trade_date_iso)
    catb = CorporateActionTimingBuffer()
    return catb.is_in_blackout(ticker_symbol, t_date)


def evaluate_institutional_delivery(
    recent_delivery_history: List[int],
    today_delivery: int,
    is_expiry_week: bool = False
) -> Dict[str, Any]:
    """
    Computes 5-day delivery volume SMA and verifies institutional accumulation:
    - Normal session: Today delivery > 1.5x 5-Day SMA
    - Expiry week: Today delivery > 1.725x 5-Day SMA
    """
    client = NSEArchiveDeliveryClient()
    return client.calculate_delivery_expansion(recent_delivery_history, today_delivery, is_expiry_week)


def calculate_risk_coordinates(entry_price: float, stop_loss: float) -> Dict[str, Any]:
    """
    Calculates exact risk per share, +2R profit target, and risk-to-reward ratio.
    """
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return {
            "is_valid": False,
            "error": f"Entry price (₹{entry_price:.2f}) must be greater than Stop-Loss (₹{stop_loss:.2f})."
        }

    target_price = round(entry_price + (2.0 * risk_per_share), 2)
    rr_ratio = round((target_price - entry_price) / risk_per_share, 2)

    return {
        "is_valid": True,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target_price": target_price,
        "risk_per_share": round(risk_per_share, 2),
        "risk_reward_ratio": rr_ratio,
        "reward_in_r": 2.0,
    }


def evaluate_exit_status(
    entry_price: float,
    stop_loss: float,
    target_price: float,
    invalidation_level: float,
    days_held: int,
    daily_candle: Dict[str, float],
    has_upcoming_earnings: bool = False
) -> Dict[str, Any]:
    """
    Evaluates open position against the 5 real-time exit rules:
    Target Hit (+2R), Stop Loss (-1R), Technical Invalidation, 15-Day Stagnation, Catalyst Emergency.
    """
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        invalidation_level=invalidation_level,
        days_held=days_held,
        daily_candle=daily_candle,
        has_upcoming_earnings=has_upcoming_earnings,
    )
    return {
        "is_exit_triggered": res.is_exit_triggered,
        "exit_reason": res.exit_reason,
        "exit_price": res.exit_price,
        "realized_r_multiple": res.realized_r_multiple,
        "holding_days_elapsed": res.holding_days_elapsed,
        "pnl_percentage": res.pnl_percentage,
        "diagnostic_notes": res.diagnostic_notes,
    }
