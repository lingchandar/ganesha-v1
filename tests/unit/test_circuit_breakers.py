"""
Unit tests for Portfolio Circuit Breakers & Correlated-Drawdown Guard (src/risk/circuit_breakers.py).
Verifies:
1. Max 5 concurrent positions cap
2. Sector concentration cap (max 2 per sector)
3. 5% aggregate capital-at-risk limit
4. Correlated-drawdown pause brake (>= 3 stop-outs in 7 days -> 7 day pause)
"""
from datetime import date, timedelta
import pytest
from src.risk.circuit_breakers import (
    PortfolioCircuitBreakers,
    CircuitBreakerCheckResult,
    CorrelatedDrawdownStatus,
)


def test_max_concurrent_positions_cap():
    # 5 active positions already open
    active_positions = [
        {"ticker_symbol": "RELIANCE", "sector": "OIL_AND_GAS", "capital_at_risk_net_inr": 4800.0},
        {"ticker_symbol": "TCS", "sector": "INFORMATION_TECHNOLOGY", "capital_at_risk_net_inr": 4900.0},
        {"ticker_symbol": "HDFCBANK", "sector": "FINANCIAL_SERVICES", "capital_at_risk_net_inr": 4700.0},
        {"ticker_symbol": "MARUTI", "sector": "AUTOMOBILE", "capital_at_risk_net_inr": 4600.0},
        {"ticker_symbol": "SUNPHARMA", "sector": "HEALTHCARE", "capital_at_risk_net_inr": 4500.0},
    ]

    res = PortfolioCircuitBreakers.evaluate_candidate_capacity(
        active_positions=active_positions,
        candidate_ticker="BHARTIARTL",
        candidate_sector="TELECOM",
        candidate_risk_inr=4800.0,
        total_capital=500000.0,
    )

    assert res.is_allowed is False
    assert "PORTFOLIO_CAPACITY_REACHED" in str(res.rejection_reason)


def test_sector_concentration_cap_max_two():
    # Already 2 IT positions open
    active_positions = [
        {"ticker_symbol": "TCS", "sector": "INFORMATION_TECHNOLOGY", "capital_at_risk_net_inr": 4900.0},
        {"ticker_symbol": "INFY", "sector": "INFORMATION_TECHNOLOGY", "capital_at_risk_net_inr": 4800.0},
    ]

    # Attempting to add a 3rd IT stock (WIPRO)
    res = PortfolioCircuitBreakers.evaluate_candidate_capacity(
        active_positions=active_positions,
        candidate_ticker="WIPRO",
        candidate_sector="INFORMATION_TECHNOLOGY",
        candidate_risk_inr=4500.0,
        total_capital=500000.0,
    )

    assert res.is_allowed is False
    assert "SECTOR_CONCENTRATION_CAP_EXCEEDED" in str(res.rejection_reason)


def test_aggregate_portfolio_risk_cap():
    # Capital: ₹500,000. Max 5% aggregate risk = ₹25,000.
    # Existing positions have ₹22,000 risk
    active_positions = [
        {"ticker_symbol": "RELIANCE", "sector": "OIL_AND_GAS", "capital_at_risk_net_inr": 11000.0},
        {"ticker_symbol": "TCS", "sector": "INFORMATION_TECHNOLOGY", "capital_at_risk_net_inr": 11000.0},
    ]

    # Candidate with ₹4,500 risk (22,000 + 4,500 = 26,500 > 25,000)
    res = PortfolioCircuitBreakers.evaluate_candidate_capacity(
        active_positions=active_positions,
        candidate_ticker="HDFCBANK",
        candidate_sector="FINANCIAL_SERVICES",
        candidate_risk_inr=4500.0,
        total_capital=500000.0,
    )

    assert res.is_allowed is False
    assert "AGGREGATE_RISK_LIMIT_EXCEEDED" in str(res.rejection_reason)


def test_correlated_drawdown_pause_triggered():
    today = date(2026, 9, 25)

    # 3 positions stopped out in trailing 7 days
    recent_closed = [
        {"ticker_symbol": "TATAMOTORS", "exit_date": date(2026, 9, 21), "exit_reason": "STOP_LOSS_HIT"},
        {"ticker_symbol": "AXISBANK", "exit_date": date(2026, 9, 23), "exit_reason": "STOP_LOSS_HIT"},
        {"ticker_symbol": "JSWSTEEL", "exit_date": date(2026, 9, 24), "exit_reason": "STOP_LOSS_HIT"},
    ]

    status = PortfolioCircuitBreakers.check_correlated_drawdown_pause(
        recent_closed_trades=recent_closed,
        current_date=today,
    )

    assert status.is_system_paused is True
    assert status.recent_stop_outs_count == 3
    assert status.pause_until_date == (date(2026, 9, 24) + timedelta(days=7)).isoformat()
    assert "CORRELATED_DRAWDOWN_PAUSE_ACTIVE" in status.status_message


def test_correlated_drawdown_normal_when_under_threshold():
    today = date(2026, 9, 25)

    # Only 1 stop-out in trailing 7 days, 1 target hit
    recent_closed = [
        {"ticker_symbol": "TATAMOTORS", "exit_date": date(2026, 9, 22), "exit_reason": "STOP_LOSS_HIT"},
        {"ticker_symbol": "INFY", "exit_date": date(2026, 9, 23), "exit_reason": "TARGET_HIT"},
    ]

    status = PortfolioCircuitBreakers.check_correlated_drawdown_pause(
        recent_closed_trades=recent_closed,
        current_date=today,
    )

    assert status.is_system_paused is False
    assert status.recent_stop_outs_count == 1
    assert status.pause_until_date is None
