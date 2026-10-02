"""Unit tests for the stateful ActiveTradeManager."""

from datetime import date

import pytest

from src.risk.active_trade_manager import ActiveTradeManager
from src.risk.exit_engine import ExitReason
from src.setups.swing_setups import SwingSetupSignal


def make_signal(symbol="NSE:TCS-EQ"):
    return SwingSetupSignal(
        ticker_symbol=symbol,
        setup_type="TREND_PULLBACK",
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        risk_reward_ratio=2.0,
        risk_per_share=10.0,
        atr_14=4.0,
        volume_surge_multiple=1.2,
        delivery_ratio=0.0,
        invalidation_level=95.0,
        rationale="test signal",
    )


def test_open_trade_from_signal():
    manager = ActiveTradeManager()
    trade = manager.open_trade(make_signal(), date(2026, 10, 1), shares=25)

    assert trade.ticker_symbol == "NSE:TCS-EQ"
    assert trade.shares == 25
    assert trade.days_held == 0
    assert manager.has_active_trade("NSE:TCS-EQ")


def test_duplicate_symbol_is_rejected():
    manager = ActiveTradeManager()
    manager.open_trade(make_signal(), date(2026, 10, 1))

    with pytest.raises(ValueError, match="ACTIVE_TRADE_EXISTS"):
        manager.open_trade(make_signal(), date(2026, 10, 1))


def test_days_held_increments_only_for_new_evaluation_date():
    manager = ActiveTradeManager()
    manager.open_trade(make_signal(), date(2026, 10, 1))

    candle = {"open": 101, "high": 104, "low": 99, "close": 103}
    first = manager.evaluate_daily_candle("NSE:TCS-EQ", date(2026, 10, 2), candle)
    second = manager.evaluate_daily_candle("NSE:TCS-EQ", date(2026, 10, 2), candle)

    assert first.holding_days_elapsed == 1
    assert second.holding_days_elapsed == 1
    assert manager.active_trades["NSE:TCS-EQ"].days_held == 1


def test_target_exit_moves_trade_to_closed_history():
    manager = ActiveTradeManager()
    manager.open_trade(make_signal(), date(2026, 10, 1))

    result = manager.evaluate_daily_candle(
        "NSE:TCS-EQ",
        date(2026, 10, 2),
        {"open": 101, "high": 121, "low": 100, "close": 118},
    )

    assert result.is_exit_triggered is True
    assert result.exit_reason == ExitReason.TARGET_HIT.value
    assert not manager.has_active_trade("NSE:TCS-EQ")
    assert manager.closed_trades["NSE:TCS-EQ"] == result


def test_evaluation_before_entry_is_rejected():
    manager = ActiveTradeManager()
    manager.open_trade(make_signal(), date(2026, 10, 5))

    with pytest.raises(ValueError, match="before entry_date"):
        manager.evaluate_daily_candle(
            "NSE:TCS-EQ",
            date(2026, 10, 4),
            {"open": 100, "high": 101, "low": 99, "close": 100},
        )


def test_close_all_like_evaluation():
    manager = ActiveTradeManager()
    manager.open_trade(make_signal("NSE:TCS-EQ"), date(2026, 10, 1))
    manager.open_trade(make_signal("NSE:INFY-EQ"), date(2026, 10, 1))

    for symbol in ("NSE:TCS-EQ", "NSE:INFY-EQ"):
        result = manager.evaluate_daily_candle(
            symbol,
            date(2026, 10, 2),
            {"open": 101, "high": 121, "low": 100, "close": 118},
        )
        assert result.exit_reason == ExitReason.TARGET_HIT.value

    assert len(manager.active_trades) == 0


def test_target_exit_creates_complete_closed_trade_record():
    manager = ActiveTradeManager()
    manager.open_trade(make_signal(), date(2026, 10, 1), shares=25)

    manager.evaluate_daily_candle(
        "NSE:TCS-EQ",
        date(2026, 10, 2),
        {"open": 101, "high": 121, "low": 100, "close": 118},
    )

    record = manager.closed_trade_records[-1]
    assert record.ticker_symbol == "NSE:TCS-EQ"
    assert record.exit_price == 120.0
    assert record.invalidation_level == 95.0
    assert record.shares == 25
    assert record.holding_days == 1
    assert record.realized_r_multiple == 2.0
    assert record.pnl_inr == 500.0
