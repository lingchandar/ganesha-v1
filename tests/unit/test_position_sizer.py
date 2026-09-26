"""
Unit tests for Friction-Adjusted Position Sizing Engine (src/risk/position_sizer.py).
Verifies exact Indian statutory delivery taxes, DP charges, slippage buffer, and 1% capital risk sizing.
"""
import pytest
from src.risk.position_sizer import (
    PositionSizingEngine,
    StatutoryFrictionBreakdown,
    PositionSizingResult,
)


def test_statutory_friction_exact_calculation():
    # Trade: 100 shares @ Entry ₹1,000, Exit ₹900
    # Buy turnover = 100 * 1000 = 100,000
    # Sell turnover = 100 * 900 = 90,000
    # Total turnover = 190,000
    friction = PositionSizingEngine.calculate_statutory_friction(
        shares=100,
        entry_price=1000.0,
        exit_price=900.0,
        brokerage_per_order=0.0
    )

    # STT: 0.1% of 100,000 + 0.1% of 90,000 = 100 + 90 = 190.0
    assert friction.stt_inr == 190.0

    # NSE exchange: 0.00297% of 190,000 = 5.643 -> 5.64
    assert friction.exchange_turnover_inr == 5.64

    # SEBI turnover: 0.0001% of 190,000 = 0.19
    assert friction.sebi_turnover_inr == 0.19

    # Stamp duty (Buy only): 0.015% of 100,000 = 15.0
    assert friction.stamp_duty_inr == 15.0

    # GST: 18% of (5.64 + 0.19) = 18% of 5.83 = 1.05
    assert friction.gst_inr == 1.05

    # DP charge: Flat ₹15.93
    assert friction.dp_charges_inr == 15.93

    # Slippage: 0.05% of 100,000 + 0.05% of 90,000 = 50 + 45 = 95.0
    assert friction.slippage_buffer_inr == 95.0

    # Total friction
    expected_total = round(190.0 + 5.64 + 0.19 + 15.0 + 1.05 + 15.93 + 95.0, 2)
    assert friction.total_friction_inr == expected_total


def test_position_sizing_respects_one_percent_risk():
    # Capital: ₹500,000. 1% Risk Budget = ₹5,000.
    # Entry: ₹2,000, Stop Loss: ₹1,900 (Risk per share = ₹100)
    res = PositionSizingEngine.calculate_position_size(
        ticker_symbol="RELIANCE",
        entry_price=2000.0,
        stop_loss=1900.0,
        total_capital=500000.0,
        max_risk_pct=0.01,
    )

    assert res.is_executable is True
    assert res.shares_to_buy > 0
    # Net capital at risk (gross loss + friction) must NOT exceed ₹5,000
    assert res.capital_at_risk_net_inr <= 5000.0
    assert res.risk_percentage_of_account <= 1.0
    assert res.target_price == 2200.0  # +2R target


def test_invalid_stop_loss_rejection():
    # Entry <= Stop Loss must be rejected
    res = PositionSizingEngine.calculate_position_size(
        ticker_symbol="TCS",
        entry_price=3500.0,
        stop_loss=3600.0,
        total_capital=500000.0,
    )
    assert res.is_executable is False
    assert res.shares_to_buy == 0
    assert "INVALID_STOP_LOSS" in str(res.rejection_reason)


def test_twenty_five_percent_position_allocation_cap():
    # If stop-loss is extremely tight (e.g. ₹1.0 risk on ₹1,000 stock),
    # 1% risk of ₹500,000 is ₹5,000 -> raw allocation would be ~5,000 shares (₹5,000,000 required!).
    # Max 25% position cap = ₹125,000 -> max 125 shares.
    res = PositionSizingEngine.calculate_position_size(
        ticker_symbol="INFY",
        entry_price=1000.0,
        stop_loss=999.0,
        total_capital=500000.0,
    )
    assert res.is_executable is True
    # Capital required must NOT exceed 25% of ₹500,000 = ₹125,000
    assert res.capital_required_inr <= 125000.0
    assert res.shares_to_buy <= 125


def test_zero_shares_when_friction_exceeds_budget():
    # Extremely small capital ₹1,000, 1% budget = ₹10
    # Flat DP charges (₹15.93) alone exceed risk budget
    res = PositionSizingEngine.calculate_position_size(
        ticker_symbol="SBIN",
        entry_price=800.0,
        stop_loss=750.0,
        total_capital=1000.0,
    )
    assert res.is_executable is False
    assert res.shares_to_buy == 0
    assert "RISK_BUDGET_TOO_SMALL" in str(res.rejection_reason)
