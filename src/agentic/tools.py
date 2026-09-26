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
from src.risk.position_sizer import PositionSizingEngine
from src.risk.circuit_breakers import PortfolioCircuitBreakers


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


def calculate_risk_envelope(
    ticker_symbol: str,
    entry_price: float,
    stop_loss: float,
    target_price: Optional[float] = None,
    total_capital: Optional[float] = None,
    max_risk_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes 1% friction-adjusted position sizing with full Indian statutory delivery taxes,
    DP charges, slippage buffer, and allocated share quantity.
    """
    res = PositionSizingEngine.calculate_position_size(
        ticker_symbol=ticker_symbol,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        total_capital=total_capital,
        max_risk_pct=max_risk_pct,
    )
    return res.to_dict()


def check_portfolio_circuit_breakers(
    active_positions: List[Dict[str, Any]],
    candidate_ticker: str,
    candidate_sector: str,
    candidate_risk_inr: float,
    total_capital: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validates candidate against:
    - 5 concurrent positions limit
    - 2 per sector concentration cap
    - 5% aggregate capital-at-risk limit
    """
    res = PortfolioCircuitBreakers.evaluate_candidate_capacity(
        active_positions=active_positions,
        candidate_ticker=candidate_ticker,
        candidate_sector=candidate_sector,
        candidate_risk_inr=candidate_risk_inr,
        total_capital=total_capital,
    )
    return res.to_dict()


def check_correlated_drawdown(
    recent_closed_trades: List[Dict[str, Any]],
    current_date: date,
) -> Dict[str, Any]:
    """
    Evaluates Correlated-Drawdown Pause Rule (>= 3 stop-outs in trailing 7 days -> 7 day pause).
    """
    status = PortfolioCircuitBreakers.check_correlated_drawdown_pause(
        recent_closed_trades=recent_closed_trades,
        current_date=current_date,
    )
    return status.to_dict()


def evaluate_sector_relative_strength(
    stock_series: pd.Series,
    sector_series: pd.Series,
    nifty_series: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates dual-layer relative strength:
    1. Sector ROC20 > 0 relative to Nifty 50
    2. Stock Mansfield RS > 0 against BOTH Nifty 50 and Sector Index
    """
    return RelativeStrengthEngine.evaluate_dual_layer_filter(
        stock_series=stock_series,
        sector_series=sector_series,
        nifty_series=nifty_series,
    )

