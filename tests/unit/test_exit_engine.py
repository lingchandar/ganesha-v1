"""
Unit tests for Precision Exit Timing Engine (src/risk/exit_engine.py).
Ensures all 5 exit conditions operate with zero emotional interference.
"""
import pytest
from src.risk.exit_engine import PrecisionExitEngine, ExitReason


def test_target_hit_plus_2r():
    # Entry: 100, Stop: 90 (Risk = 10), Target: 120 (+2R), Invalidation: 95
    candle = {"open": 105, "high": 121, "low": 104, "close": 118}
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        invalidation_level=95.0,
        days_held=4,
        daily_candle=candle
    )
    assert res.is_exit_triggered is True
    assert res.exit_reason == ExitReason.TARGET_HIT.value
    assert res.exit_price == 120.0
    assert res.realized_r_multiple == 2.0


def test_stop_loss_hit_minus_1r():
    candle = {"open": 98, "high": 99, "low": 89, "close": 91}
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        invalidation_level=95.0,
        days_held=3,
        daily_candle=candle
    )
    assert res.is_exit_triggered is True
    assert res.exit_reason == ExitReason.STOP_LOSS_HIT.value
    assert res.exit_price == 90.0
    assert res.realized_r_multiple == -1.0


def test_overnight_gap_down_realism():
    # Stock gaps down directly to 85 (below stop-loss 90)
    candle = {"open": 85, "high": 88, "low": 84, "close": 86}
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        invalidation_level=95.0,
        days_held=2,
        daily_candle=candle
    )
    assert res.is_exit_triggered is True
    assert res.exit_reason == ExitReason.OVERNIGHT_GAP_STOP.value
    assert res.exit_price == 85.0  # Real market open, not 90!
    assert res.realized_r_multiple == -1.5  # (85 - 100) / 10 = -1.5R


def test_technical_invalidation_exit():
    # Day close breaks invalidation anchor 95, but didn't touch SL 90
    candle = {"open": 97, "high": 97, "low": 93, "close": 94}
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        invalidation_level=95.0,
        days_held=5,
        daily_candle=candle
    )
    assert res.is_exit_triggered is True
    assert res.exit_reason == ExitReason.TECHNICAL_INVALIDATION.value
    assert res.exit_price == 94.0
    assert res.realized_r_multiple == -0.6  # (94 - 100) / 10 = -0.6R


def test_time_exit_15_day_stagnation():
    # Day 15 reached, wandering inside 96-104 without hitting SL or Target
    candle = {"open": 102, "high": 104, "low": 101, "close": 103}
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        invalidation_level=95.0,
        days_held=15,
        daily_candle=candle
    )
    assert res.is_exit_triggered is True
    assert res.exit_reason == ExitReason.TIME_EXIT_STAGNATION.value
    assert res.exit_price == 103.0
    assert res.realized_r_multiple == 0.3  # (103 - 100) / 10 = +0.3R


def test_position_still_active_on_day_6():
    candle = {"open": 103, "high": 107, "low": 102, "close": 106}
    res = PrecisionExitEngine.evaluate_active_trade(
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        invalidation_level=95.0,
        days_held=6,
        daily_candle=candle
    )
    assert res.is_exit_triggered is False
    assert res.exit_reason == ExitReason.STILL_OPEN.value
    assert res.exit_price == 106.0
