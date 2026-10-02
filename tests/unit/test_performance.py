"""
Focused tests for runtime closed-trade performance adaptation.
"""

from datetime import date

from src.backtest.performance import BacktestPerformanceAnalyzer, DailyEquityPoint
from src.risk.active_trade_manager import ClosedTrade


def make_closed_trade():
    return ClosedTrade(
        ticker_symbol="NSE:TCS-EQ",
        setup_type="TREND_PULLBACK",
        entry_date=date(2026, 10, 1),
        exit_date=date(2026, 10, 3),
        entry_price=100.0,
        exit_price=110.0,
        stop_loss=95.0,
        target_price=110.0,
        invalidation_level=96.0,
        shares=10,
        holding_days=2,
        exit_reason="TARGET_HIT_2R",
        realized_r_multiple=2.0,
        pnl_percentage=10.0,
        pnl_inr=100.0,
    )


def test_closed_trade_adapter_preserves_invalidation_and_friction():
    trade = BacktestPerformanceAnalyzer.closed_trade_to_backtest_trade(
        make_closed_trade(),
        trade_id="closed-000001",
        sector="IT",
        entry_friction_inr=2.5,
        exit_friction_inr=3.5,
    )

    assert trade.trade_id == "closed-000001"
    assert trade.sector == "IT"
    assert trade.invalidation_level == 96.0
    assert trade.gross_pnl_inr == 100.0
    assert trade.total_friction_inr == 6.0
    assert trade.net_pnl_inr == 94.0
    assert trade.net_return_pct == 9.4


def test_closed_trade_batch_conversion_and_analysis():
    closed = [make_closed_trade()]
    equity = [
        DailyEquityPoint(date(2026, 10, 1), 1000.0, 0.0, 1000.0, 0, 0.0),
        DailyEquityPoint(date(2026, 10, 3), 1094.0, 0.0, 1094.0, 0, 0.0),
    ]

    trades = BacktestPerformanceAnalyzer.closed_trades_to_backtest_trades(
        closed,
        sector_by_symbol={"NSE:TCS-EQ": "IT"},
        frictions_by_trade_id={"closed-000001": (2.0, 2.0)},
    )

    assert len(trades) == 1
    assert trades[0].trade_id == "closed-000001"
    assert trades[0].sector == "IT"
    assert trades[0].net_pnl_inr == 96.0

    summary = BacktestPerformanceAnalyzer.analyze_closed_trades(
        closed_trades=closed,
        equity_curve=equity,
        initial_capital=1000.0,
        sector_by_symbol={"NSE:TCS-EQ": "IT"},
        frictions_by_trade_id={"closed-000001": (2.0, 2.0)},
    )

    assert summary.total_trades == 1
    assert summary.winning_trades == 1
    assert summary.net_profit_inr == 94.0
    assert summary.total_friction_inr == 4.0
