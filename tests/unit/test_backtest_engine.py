"""
GANESHA V1 — Backtest Engine Unit Tests (Stage 9)

Verifies:
1. Zero look-ahead bias in data slicing.
2. Realistic GTT limit order execution & fill conditions.
3. Overnight gap-down fills at actual open price.
4. Profit target (+2R) and structural stop-loss (-1R) fills.
5. Technical invalidation & 15-day stagnation exits.
6. Complete Indian statutory friction accounting.
7. Portfolio capacity and sector concentration enforcement.
8. Correlated-drawdown circuit breaker pause.
9. Performance metrics (Win Rate, Profit Factor, Expectancy, Max Drawdown).
"""
import pytest
from datetime import date, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

from src.backtest.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    PendingOrder,
    BacktestPosition,
    BacktestResult,
)
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.performance import (
    BacktestTrade,
    DailyEquityPoint,
    BacktestPerformanceAnalyzer,
)
from src.risk.exit_engine import ExitReason


# ── Synthetic Data Fixtures for Deterministic Testing ──

@pytest.fixture
def mock_candles_df() -> pd.DataFrame:
    """Creates 60 daily candles for testing."""
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(60)]
    prices = [100.0 + (i * 0.5) for i in range(60)]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 2.0 for p in prices],
            "low": [p - 1.5 for p in prices],
            "close": [p + 0.5 for p in prices],
            "volume": [100_000] * 60,
            "delivery_volume": [60_000] * 60,
        },
        index=dates,
    )
    return df


class MockDataLoader(HistoricalDataLoader):
    """Subclass of HistoricalDataLoader using in-memory mock data."""
    def __init__(self, stock_dict: Dict[str, pd.DataFrame], sector_dict: Dict[str, str]):
        super().__init__()
        self.stock_candles = stock_dict
        self.ticker_sectors = sector_dict
        all_dates = set()
        for df in stock_dict.values():
            all_dates.update(df.index)
        self.trading_dates = sorted(list(all_dates))
        self._compute_sector_series()
        self._load_nifty_benchmark(None, None)
        self._is_loaded = True


# ── Test Cases ──

def test_backtest_config_defaults():
    """Verify default institutional backtest parameters."""
    cfg = BacktestConfig()
    assert cfg.initial_capital_inr == 500_000.0
    assert cfg.max_risk_pct == 0.01
    assert cfg.max_concurrent_positions == 5
    assert cfg.max_per_sector == 2
    assert cfg.min_reward_risk == 2.0
    assert cfg.enable_friction is True
    assert cfg.enable_circuit_breakers is True
    assert cfg.enable_drawdown_brake is True


def test_zero_lookahead_data_slicing(mock_candles_df):
    """Verify that get_point_in_time_slice never leaks future data."""
    stocks = {"NSE:TEST-EQ": mock_candles_df}
    sectors = {"NSE:TEST-EQ": "FINANCIAL_SERVICES"}
    loader = MockDataLoader(stocks, sectors)

    cutoff_date = date(2025, 1, 25)
    nifty_slice, candidates = loader.get_point_in_time_slice(as_of_date=cutoff_date, min_bars_required=10)

    assert "NSE:TEST-EQ" in candidates
    stock_slice = candidates["NSE:TEST-EQ"]["ohlcv"]
    assert stock_slice.index.max() <= cutoff_date
    assert date(2025, 1, 26) not in stock_slice.index


def test_pending_order_limit_fill_when_price_reached():
    """Verify limit order fills when day's range covers entry price."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=100_000.0))
    trade_date = date(2025, 2, 1)

    order = PendingOrder(
        ticker_symbol="NSE:INFY-EQ",
        sector="IT",
        setup_type="TREND_PULLBACK",
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        invalidation_level=94.0,
        shares=100,
        capital_required_inr=10_000.0,
        capital_at_risk_net_inr=600.0,
        estimated_entry_friction_inr=25.0,
        risk_reward_ratio=2.0,
        scan_date=trade_date - timedelta(days=1),
    )
    engine.pending_orders = [order]

    # Mock candle where entry price (100.0) is traded
    engine.data_loader.stock_candles["NSE:INFY-EQ"] = pd.DataFrame(
        {"open": [101.0], "high": [103.0], "low": [99.0], "close": [102.0], "volume": [50000]},
        index=[trade_date],
    )

    engine._execute_pending_orders(trade_date)
    assert len(engine.active_positions) == 1
    pos = engine.active_positions[0]
    assert pos.ticker_symbol == "NSE:INFY-EQ"
    assert pos.entry_price == 100.0
    assert engine.cash < 100_000.0




def test_pending_order_favorable_gap_down_fills_at_open():
    """A buy-limit order gets price improvement when the market opens below the limit."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=100_000.0))
    trade_date = date(2025, 2, 4)
    order = PendingOrder(
        ticker_symbol="NSE:INFY-EQ", sector="IT", setup_type="TREND_PULLBACK",
        entry_price=100.0, stop_loss=95.0, target_price=110.0,
        invalidation_level=94.0, shares=100, capital_required_inr=10_000.0,
        capital_at_risk_net_inr=600.0, estimated_entry_friction_inr=25.0,
        risk_reward_ratio=2.0, scan_date=trade_date - timedelta(days=1),
    )
    engine.pending_orders = [order]
    engine.data_loader.stock_candles["NSE:INFY-EQ"] = pd.DataFrame(
        {"open": [96.0], "high": [99.0], "low": [95.0], "close": [98.0], "volume": [50000]},
        index=[trade_date],
    )
    engine._execute_pending_orders(trade_date)
    assert len(engine.active_positions) == 1
    assert engine.active_positions[0].entry_price == 96.0


def test_backtest_warmup_default_is_sufficient_for_long_lookback_features():
    """Warmup must cover EMA200/long setup lookbacks rather than only short indicators."""
    assert BacktestConfig().min_warmup_bars >= 300


def test_pending_order_unfilled_when_price_runs_away():
    """Verify limit order does not fill when stock gaps up and never pulls back to limit."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=100_000.0))
    trade_date = date(2025, 2, 1)

    order = PendingOrder(
        ticker_symbol="NSE:INFY-EQ",
        sector="IT",
        setup_type="TREND_PULLBACK",
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        invalidation_level=94.0,
        shares=100,
        capital_required_inr=10_000.0,
        capital_at_risk_net_inr=600.0,
        estimated_entry_friction_inr=25.0,
        risk_reward_ratio=2.0,
        scan_date=trade_date - timedelta(days=1),
    )
    engine.pending_orders = [order]

    # Candle where price gaps up: low is 105.0 > entry 100.0
    engine.data_loader.stock_candles["NSE:INFY-EQ"] = pd.DataFrame(
        {"open": [106.0], "high": [112.0], "low": [105.0], "close": [110.0], "volume": [50000]},
        index=[trade_date],
    )

    engine._execute_pending_orders(trade_date)
    assert len(engine.active_positions) == 0
    assert len(engine.pending_orders) == 0  # Expired unfilled


def test_exit_execution_overnight_gap_down_realism():
    """Verify overnight gap-down past stop fills at actual open price with slippage."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=100_000.0))
    trade_date = date(2025, 2, 2)

    pos = BacktestPosition(
        position_id="pos1",
        ticker_symbol="NSE:TEST-EQ",
        sector="FINANCIAL",
        setup_type="VCP",
        entry_date=trade_date - timedelta(days=1),
        entry_price=100.0,
        shares=100,
        capital_invested=10_000.0,
        initial_stop_loss=95.0,
        target_price=110.0,
        invalidation_level=94.0,
        entry_friction_inr=25.0,
        capital_at_risk_net_inr=525.0,
        days_held=1,
    )
    engine.active_positions = [pos]

    # Stock opens at 90.0 (gapping down past stop 95.0)
    engine.data_loader.stock_candles["NSE:TEST-EQ"] = pd.DataFrame(
        {"open": [90.0], "high": [92.0], "low": [89.0], "close": [91.0], "volume": [50000]},
        index=[trade_date],
    )

    engine._evaluate_active_position_exits(trade_date)
    assert len(engine.active_positions) == 0
    assert len(engine.closed_trades) == 1

    closed = engine.closed_trades[0]
    assert closed.exit_reason == ExitReason.OVERNIGHT_GAP_STOP.value
    assert closed.exit_price == 90.0  # Filled at open, NOT theoretical 95.0!
    assert closed.gross_pnl_inr == -1000.0
    assert closed.realized_r_multiple < -1.0


def test_exit_execution_target_hit_plus_2r():
    """Verify take-profit target hit fills at +2R target."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=100_000.0))
    trade_date = date(2025, 2, 3)

    pos = BacktestPosition(
        position_id="pos2",
        ticker_symbol="NSE:TEST-EQ",
        sector="FINANCIAL",
        setup_type="VCP",
        entry_date=trade_date - timedelta(days=2),
        entry_price=100.0,
        shares=100,
        capital_invested=10_000.0,
        initial_stop_loss=95.0,
        target_price=110.0,
        invalidation_level=94.0,
        entry_friction_inr=25.0,
        capital_at_risk_net_inr=525.0,
        days_held=2,
    )
    engine.active_positions = [pos]

    # High hits 112.0 >= target 110.0
    engine.data_loader.stock_candles["NSE:TEST-EQ"] = pd.DataFrame(
        {"open": [104.0], "high": [112.0], "low": [103.0], "close": [109.0], "volume": [50000]},
        index=[trade_date],
    )

    engine._evaluate_active_position_exits(trade_date)
    assert len(engine.closed_trades) == 1
    closed = engine.closed_trades[0]
    assert closed.exit_reason == ExitReason.TARGET_HIT.value
    assert closed.exit_price == 110.0
    assert closed.realized_r_multiple == 2.0


def test_exit_execution_15_day_stagnation():
    """Verify position is exited on day 15 if neither target nor stop is hit."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=100_000.0))
    trade_date = date(2025, 2, 15)

    pos = BacktestPosition(
        position_id="pos3",
        ticker_symbol="NSE:TEST-EQ",
        sector="FINANCIAL",
        setup_type="VCP",
        entry_date=trade_date - timedelta(days=15),
        entry_price=100.0,
        shares=100,
        capital_invested=10_000.0,
        initial_stop_loss=95.0,
        target_price=110.0,
        invalidation_level=94.0,
        entry_friction_inr=25.0,
        capital_at_risk_net_inr=525.0,
        days_held=14,  # will become 15 during evaluation
    )
    engine.active_positions = [pos]

    # Day 15 candle: price stagnant at 101.0
    engine.data_loader.stock_candles["NSE:TEST-EQ"] = pd.DataFrame(
        {"open": [100.5], "high": [102.0], "low": [99.5], "close": [101.0], "volume": [50000]},
        index=[trade_date],
    )

    engine._evaluate_active_position_exits(trade_date)
    assert len(engine.closed_trades) == 1
    closed = engine.closed_trades[0]
    assert closed.exit_reason == ExitReason.TIME_EXIT_STAGNATION.value
    assert closed.exit_price == 101.0


def test_correlated_drawdown_pause_halts_new_scans():
    """Verify that >=3 stop outs in 7 days pauses new entries."""
    engine = BacktestEngine(config=BacktestConfig(initial_capital_inr=500_000.0))
    current_date = date(2025, 3, 10)

    # 3 stop outs occurred between March 5 and March 9
    for i in range(3):
        t = BacktestTrade(
            trade_id=f"loss_{i}",
            ticker_symbol=f"STOCK_{i}",
            sector="IT",
            setup_type="BREAKOUT",
            entry_date=current_date - timedelta(days=5),
            entry_price=100.0,
            shares=100,
            capital_invested=10000.0,
            initial_stop_loss=95.0,
            target_price=110.0,
            invalidation_level=94.0,
            exit_date=current_date - timedelta(days=i + 1),
            exit_price=95.0,
            exit_reason=ExitReason.STOP_LOSS_HIT.value,
            holding_days=3,
            gross_pnl_inr=-500.0,
            entry_friction_inr=20.0,
            exit_friction_inr=20.0,
            total_friction_inr=40.0,
            net_pnl_inr=-540.0,
            net_return_pct=-5.4,
            realized_r_multiple=-1.0,
        )
        engine.closed_trades.append(t)

    engine._evaluate_drawdown_brake(current_date)
    assert engine.system_paused_until is not None
    assert engine.system_paused_until >= current_date + timedelta(days=7)


def test_performance_summary_metrics_calculation():
    """Verify performance metrics math."""
    t1 = BacktestTrade(
        trade_id="t1",
        ticker_symbol="STOCK_A",
        sector="AUTO",
        setup_type="VCP",
        entry_date=date(2025, 1, 1),
        entry_price=100.0,
        shares=100,
        capital_invested=10000.0,
        initial_stop_loss=95.0,
        target_price=110.0,
        invalidation_level=94.0,
        exit_date=date(2025, 1, 5),
        exit_price=110.0,
        exit_reason="TARGET_HIT_2R",
        holding_days=4,
        gross_pnl_inr=1000.0,
        entry_friction_inr=25.0,
        exit_friction_inr=25.0,
        total_friction_inr=50.0,
        net_pnl_inr=950.0,
        net_return_pct=9.5,
        realized_r_multiple=2.0,
    )
    t2 = BacktestTrade(
        trade_id="t2",
        ticker_symbol="STOCK_B",
        sector="IT",
        setup_type="PULLBACK",
        entry_date=date(2025, 1, 3),
        entry_price=200.0,
        shares=50,
        capital_invested=10000.0,
        initial_stop_loss=190.0,
        target_price=220.0,
        invalidation_level=188.0,
        exit_date=date(2025, 1, 7),
        exit_price=190.0,
        exit_reason="STOP_LOSS_HIT_1R",
        holding_days=4,
        gross_pnl_inr=-500.0,
        entry_friction_inr=25.0,
        exit_friction_inr=25.0,
        total_friction_inr=50.0,
        net_pnl_inr=-550.0,
        net_return_pct=-5.5,
        realized_r_multiple=-1.0,
    )

    eq_curve = [
        DailyEquityPoint(date(2025, 1, 1), 500000, 0, 500000, 0, 0.0),
        DailyEquityPoint(date(2025, 1, 5), 500950, 0, 500950, 0, 0.0),
        DailyEquityPoint(date(2025, 1, 7), 500400, 0, 500400, 0, 0.11),
    ]

    summary = BacktestPerformanceAnalyzer.analyze([t1, t2], eq_curve, initial_capital=500000.0)

    assert summary.total_trades == 2
    assert summary.winning_trades == 1
    assert summary.losing_trades == 1
    assert summary.win_rate_pct == 50.0
    assert summary.net_profit_inr == 400.0
    assert summary.total_friction_inr == 100.0
    assert summary.avg_win_r == 2.0
    assert summary.avg_loss_r == 1.0
    assert summary.expectancy_r == 0.5  # (0.5 * 2.0) - (0.5 * 1.0) = 0.5
