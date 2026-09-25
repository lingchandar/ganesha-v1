"""
GANESHA V1 — Precision Exit Timing Engine (Module 10, 11, 13)

REAL-TIME TRADING PHILOSOPHY:
"Entry determines your risk; Exit determines your profitability."
In real-time trading, sloppy exits destroy edge. This engine enforces the 5
immutable exit conditions of GANESHA V1 without emotional interference.

The 5 Exit Conditions:
1. TARGET_HIT (+2R): Price touches or exceeds target price (Entry + 2 * Risk).
2. STOP_LOSS_HIT (-1R): Price touches or breaks structural stop loss.
   (Includes overnight gap-down realism: exits at Open price if gap below SL).
3. TECHNICAL_INVALIDATION: Daily close violates setup invalidation anchor (e.g. close < 20 EMA).
4. TIME_EXIT_STAGNATION: Holding period reaches 15 trading days without hitting +2R or -1R.
   (Liquidates position at Day 15 close to release stagnant capital).
5. CATALYST_EMERGENCY_EXIT: Upcoming scheduled corporate earnings or board meeting.
"""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, Optional
import pandas as pd
from loguru import logger


class ExitReason(str, Enum):
    TARGET_HIT = "TARGET_HIT (+2R)"
    STOP_LOSS_HIT = "STOP_LOSS_HIT (-1R)"
    OVERNIGHT_GAP_STOP = "OVERNIGHT_GAP_STOP (-1R+)"
    TECHNICAL_INVALIDATION = "TECHNICAL_INVALIDATION"
    TIME_EXIT_STAGNATION = "TIME_EXIT_STAGNATION (15-DAY LIMIT)"
    CATALYST_EMERGENCY_EXIT = "CATALYST_EMERGENCY_EXIT"
    STILL_OPEN = "POSITION_ACTIVE"


@dataclass
class ExitEvaluationResult:
    is_exit_triggered: bool
    exit_reason: str
    exit_price: float
    realized_r_multiple: float
    holding_days_elapsed: int
    pnl_percentage: float
    diagnostic_notes: str


class PrecisionExitEngine:
    """
    Evaluates active swing positions against the 5 real-time exit rules.
    """

    MAX_HOLDING_DAYS = 15

    @classmethod
    def evaluate_active_trade(
        cls,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        invalidation_level: float,
        days_held: int,
        daily_candle: Dict[str, float],
        has_upcoming_earnings: bool = False,
    ) -> ExitEvaluationResult:
        """
        Evaluate a daily session candle for an open swing trade.

        Args:
            entry_price: Initial execution fill price
            stop_loss: Initial structural stop-loss price
            target_price: +2R target price
            invalidation_level: Technical invalidation price level
            days_held: Trading days elapsed since entry (1-indexed)
            daily_candle: Dict with keys: ['open', 'high', 'low', 'close']
            has_upcoming_earnings: True if earnings announcement announced in next 5 sessions

        Returns:
            ExitEvaluationResult detailing exit status, fill price, and R-multiple realized.
        """
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            raise ValueError(f"Invalid risk: entry {entry_price} <= stop_loss {stop_loss}")

        c_open = daily_candle["open"]
        c_high = daily_candle["high"]
        c_low = daily_candle["low"]
        c_close = daily_candle["close"]

        # ── 1. Catalyst Emergency Exit ──
        if has_upcoming_earnings:
            realized_r = round((c_close - entry_price) / risk_per_share, 2)
            pnl_pct = round(((c_close - entry_price) / entry_price) * 100.0, 2)
            return ExitEvaluationResult(
                is_exit_triggered=True,
                exit_reason=ExitReason.CATALYST_EMERGENCY_EXIT.value,
                exit_price=c_close,
                realized_r_multiple=realized_r,
                holding_days_elapsed=days_held,
                pnl_percentage=pnl_pct,
                diagnostic_notes="Emergency exit triggered: high corporate catalyst event risk within holding horizon.",
            )

        # ── 2. Overnight Gap Down Past Stop Loss ──
        # Realism: If stock opened lower than stop-loss, you get filled at the open, NOT the theoretical stop!
        if c_open <= stop_loss:
            realized_r = round((c_open - entry_price) / risk_per_share, 2)
            pnl_pct = round(((c_open - entry_price) / entry_price) * 100.0, 2)
            return ExitEvaluationResult(
                is_exit_triggered=True,
                exit_reason=ExitReason.OVERNIGHT_GAP_STOP.value,
                exit_price=c_open,
                realized_r_multiple=realized_r,
                holding_days_elapsed=days_held,
                pnl_percentage=pnl_pct,
                diagnostic_notes=f"Overnight gap-down open at ₹{c_open:.2f} breached stop ₹{stop_loss:.2f}.",
            )

        # ── 3. Intraday Stop Loss Hit (-1R) ──
        if c_low <= stop_loss:
            realized_r = -1.00
            pnl_pct = round(((stop_loss - entry_price) / entry_price) * 100.0, 2)
            return ExitEvaluationResult(
                is_exit_triggered=True,
                exit_reason=ExitReason.STOP_LOSS_HIT.value,
                exit_price=stop_loss,
                realized_r_multiple=realized_r,
                holding_days_elapsed=days_held,
                pnl_percentage=pnl_pct,
                diagnostic_notes=f"Structural stop-loss executed at ₹{stop_loss:.2f}.",
            )

        # ── 4. Target Price Reached (+2R Take-Profit) ──
        if c_high >= target_price:
            realized_r = 2.00
            pnl_pct = round(((target_price - entry_price) / entry_price) * 100.0, 2)
            return ExitEvaluationResult(
                is_exit_triggered=True,
                exit_reason=ExitReason.TARGET_HIT.value,
                exit_price=target_price,
                realized_r_multiple=realized_r,
                holding_days_elapsed=days_held,
                pnl_percentage=pnl_pct,
                diagnostic_notes=f"Profit target hit at ₹{target_price:.2f} (+2.00R).",
            )

        # ── 5. Technical Invalidation on Daily Close ──
        if c_close < invalidation_level:
            realized_r = round((c_close - entry_price) / risk_per_share, 2)
            pnl_pct = round(((c_close - entry_price) / entry_price) * 100.0, 2)
            return ExitEvaluationResult(
                is_exit_triggered=True,
                exit_reason=ExitReason.TECHNICAL_INVALIDATION.value,
                exit_price=c_close,
                realized_r_multiple=realized_r,
                holding_days_elapsed=days_held,
                pnl_percentage=pnl_pct,
                diagnostic_notes=f"Daily close ₹{c_close:.2f} broke invalidation anchor ₹{invalidation_level:.2f}.",
            )

        # ── 6. 15-Day Stagnation Rule (Time-Exit) ──
        if days_held >= cls.MAX_HOLDING_DAYS:
            realized_r = round((c_close - entry_price) / risk_per_share, 2)
            pnl_pct = round(((c_close - entry_price) / entry_price) * 100.0, 2)
            return ExitEvaluationResult(
                is_exit_triggered=True,
                exit_reason=ExitReason.TIME_EXIT_STAGNATION.value,
                exit_price=c_close,
                realized_r_multiple=realized_r,
                holding_days_elapsed=days_held,
                pnl_percentage=pnl_pct,
                diagnostic_notes=(
                    f"15 trading sessions elapsed without reaching +2R or -1R. "
                    f"Position liquidated at Day 15 close (₹{c_close:.2f}) to eliminate dead money."
                ),
            )

        # ── Position Remains Active ──
        unrealized_r = round((c_close - entry_price) / risk_per_share, 2)
        unrealized_pnl_pct = round(((c_close - entry_price) / entry_price) * 100.0, 2)
        return ExitEvaluationResult(
            is_exit_triggered=False,
            exit_reason=ExitReason.STILL_OPEN.value,
            exit_price=c_close,
            realized_r_multiple=unrealized_r,
            holding_days_elapsed=days_held,
            pnl_percentage=unrealized_pnl_pct,
            diagnostic_notes=f"Position active on Day {days_held}. Current price: ₹{c_close:.2f}.",
        )
