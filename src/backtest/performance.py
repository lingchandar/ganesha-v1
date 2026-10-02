"""
GANESHA V1 — Backtest Performance Analyzer

Calculates comprehensive trading performance, risk metrics, drawdown stats,
expectancy, and statutory friction breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestTrade:
    """Represents a fully executed and closed swing trade in backtesting."""
    trade_id: str
    ticker_symbol: str
    sector: str
    setup_type: str
    entry_date: date
    entry_price: float
    shares: int
    capital_invested: float
    initial_stop_loss: float
    target_price: float
    invalidation_level: float
    exit_date: date
    exit_price: float
    exit_reason: str
    holding_days: int
    gross_pnl_inr: float
    entry_friction_inr: float
    exit_friction_inr: float
    total_friction_inr: float
    net_pnl_inr: float
    net_return_pct: float
    realized_r_multiple: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_date"] = self.entry_date.isoformat()
        d["exit_date"] = self.exit_date.isoformat()
        return d


@dataclass
class DailyEquityPoint:
    """Point on the daily equity curve."""
    trade_date: date
    cash_inr: float
    invested_capital_inr: float
    total_equity_inr: float
    open_positions_count: int
    drawdown_pct: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trade_date"] = self.trade_date.isoformat()
        return d


@dataclass
class PerformanceSummary:
    """Aggregated performance metrics."""
    initial_capital_inr: float
    final_equity_inr: float
    net_profit_inr: float
    net_return_pct: float
    cagr_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    scratch_trades: int
    win_rate_pct: float
    loss_rate_pct: float

    gross_profit_inr: float
    gross_loss_inr: float
    total_friction_inr: float
    profit_factor: float
    payoff_ratio: float

    avg_trade_pnl_inr: float
    avg_win_inr: float
    avg_loss_inr: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float
    expectancy_inr: float

    max_drawdown_inr: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float

    avg_holding_days_all: float
    avg_holding_days_winners: float
    avg_holding_days_losers: float

    setup_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exit_reason_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sector_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BacktestPerformanceAnalyzer:
    """Computes statistical and institutional risk metrics."""

    @staticmethod
    def closed_trade_to_backtest_trade(
        trade,
        trade_id: str,
        sector: str = "UNKNOWN",
        entry_friction_inr: float = 0.0,
        exit_friction_inr: float = 0.0,
    ) -> BacktestTrade:
        """Adapt a live ClosedTrade into the common performance schema."""
        capital_invested = trade.entry_price * trade.shares
        gross_pnl = (trade.exit_price - trade.entry_price) * trade.shares
        total_friction = entry_friction_inr + exit_friction_inr
        net_pnl = gross_pnl - total_friction
        net_return_pct = (net_pnl / capital_invested) * 100.0 if capital_invested > 0 else 0.0
        return BacktestTrade(
            trade_id=trade_id,
            ticker_symbol=trade.ticker_symbol,
            sector=sector,
            setup_type=trade.setup_type,
            entry_date=trade.entry_date,
            entry_price=trade.entry_price,
            shares=trade.shares,
            capital_invested=round(capital_invested, 2),
            initial_stop_loss=trade.stop_loss,
            target_price=trade.target_price,
            invalidation_level=float(trade.invalidation_level),
            exit_date=trade.exit_date,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            holding_days=trade.holding_days,
            gross_pnl_inr=round(gross_pnl, 2),
            entry_friction_inr=round(entry_friction_inr, 2),
            exit_friction_inr=round(exit_friction_inr, 2),
            total_friction_inr=round(total_friction, 2),
            net_pnl_inr=round(net_pnl, 2),
            net_return_pct=round(net_return_pct, 2),
            realized_r_multiple=trade.realized_r_multiple,
        )

    @classmethod
    def closed_trades_to_backtest_trades(
        cls,
        closed_trades: List[Any],
        sector_by_symbol: Optional[Dict[str, str]] = None,
        frictions_by_trade_id: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> List[BacktestTrade]:
        """Convert closed runtime trades into the common backtest schema."""
        sector_by_symbol = sector_by_symbol or {}
        frictions_by_trade_id = frictions_by_trade_id or {}

        converted: List[BacktestTrade] = []
        for index, trade in enumerate(closed_trades, start=1):
            trade_id = f"closed-{index:06d}"
            entry_friction, exit_friction = frictions_by_trade_id.get(
                trade_id, (0.0, 0.0)
            )
            converted.append(
                cls.closed_trade_to_backtest_trade(
                    trade=trade,
                    trade_id=trade_id,
                    sector=sector_by_symbol.get(trade.ticker_symbol, "UNKNOWN"),
                    entry_friction_inr=entry_friction,
                    exit_friction_inr=exit_friction,
                )
            )
        return converted

    @classmethod
    def analyze_closed_trades(
        cls,
        closed_trades: List[Any],
        equity_curve: List[DailyEquityPoint],
        initial_capital: float,
        sector_by_symbol: Optional[Dict[str, str]] = None,
        frictions_by_trade_id: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> PerformanceSummary:
        """Analyze runtime closed trades using the standard performance engine."""
        trades = cls.closed_trades_to_backtest_trades(
            closed_trades=closed_trades,
            sector_by_symbol=sector_by_symbol,
            frictions_by_trade_id=frictions_by_trade_id,
        )
        return cls.analyze(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
        )

    """
    Computes statistical and institutional risk metrics from closed trades
    and daily equity curve snapshots.
    """

    RISK_FREE_RATE_ANNUAL = 0.065  # 6.5% Indian 10Y G-Sec / Repo benchmark

    @classmethod
    def analyze(
        cls,
        trades: List[BacktestTrade],
        equity_curve: List[DailyEquityPoint],
        initial_capital: float,
    ) -> PerformanceSummary:
        """
        Computes the complete suite of trading performance metrics.
        """
        final_equity = equity_curve[-1].total_equity_inr if equity_curve else initial_capital
        net_profit = final_equity - initial_capital
        net_return_pct = (net_profit / initial_capital) * 100.0 if initial_capital > 0 else 0.0

        # Duration & CAGR
        if len(equity_curve) > 1:
            total_calendar_days = (equity_curve[-1].trade_date - equity_curve[0].trade_date).days
            years = max(total_calendar_days / 365.25, 0.01)
            cagr_pct = (((final_equity / initial_capital) ** (1.0 / years)) - 1.0) * 100.0
        else:
            cagr_pct = 0.0

        if not trades:
            return PerformanceSummary(
                initial_capital_inr=initial_capital,
                final_equity_inr=final_equity,
                net_profit_inr=0.0,
                net_return_pct=0.0,
                cagr_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                scratch_trades=0,
                win_rate_pct=0.0,
                loss_rate_pct=0.0,
                gross_profit_inr=0.0,
                gross_loss_inr=0.0,
                total_friction_inr=0.0,
                profit_factor=0.0,
                payoff_ratio=0.0,
                avg_trade_pnl_inr=0.0,
                avg_win_inr=0.0,
                avg_loss_inr=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
                expectancy_r=0.0,
                expectancy_inr=0.0,
                max_drawdown_inr=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_duration_days=0,
                calmar_ratio=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                avg_holding_days_all=0.0,
                avg_holding_days_winners=0.0,
                avg_holding_days_losers=0.0,
            )

        total_trades = len(trades)
        wins = [t for t in trades if t.net_pnl_inr > 0]
        losses = [t for t in trades if t.net_pnl_inr < 0]
        scratches = [t for t in trades if t.net_pnl_inr == 0]

        win_count = len(wins)
        loss_count = len(losses)
        scratch_count = len(scratches)
        win_rate = (win_count / total_trades) * 100.0
        loss_rate = (loss_count / total_trades) * 100.0

        gross_profit = sum(t.gross_pnl_inr for t in wins)
        gross_loss = abs(sum(t.gross_pnl_inr for t in losses))
        total_friction = sum(t.total_friction_inr for t in trades)

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        avg_win_inr = (sum(t.net_pnl_inr for t in wins) / win_count) if win_count > 0 else 0.0
        avg_loss_inr = abs(sum(t.net_pnl_inr for t in losses) / loss_count) if loss_count > 0 else 0.0
        payoff_ratio = (avg_win_inr / avg_loss_inr) if avg_loss_inr > 0 else 0.0

        avg_win_r = (sum(t.realized_r_multiple for t in wins) / win_count) if win_count > 0 else 0.0
        avg_loss_r = abs(sum(t.realized_r_multiple for t in losses) / loss_count) if loss_count > 0 else 0.0
        expectancy_r = ((win_rate / 100.0) * avg_win_r) - ((loss_rate / 100.0) * avg_loss_r)
        expectancy_inr = net_profit / total_trades

        # Holding duration
        avg_hold_all = sum(t.holding_days for t in trades) / total_trades
        avg_hold_win = (sum(t.holding_days for t in wins) / win_count) if win_count > 0 else 0.0
        avg_hold_loss = (sum(t.holding_days for t in losses) / loss_count) if loss_count > 0 else 0.0

        # Drawdown analysis on equity curve
        eq_series = pd.Series([pt.total_equity_inr for pt in equity_curve])
        cum_max = eq_series.cummax()
        drawdown_inr_series = cum_max - eq_series
        drawdown_pct_series = (drawdown_inr_series / cum_max) * 100.0

        max_dd_inr = float(drawdown_inr_series.max()) if not drawdown_inr_series.empty else 0.0
        max_dd_pct = float(drawdown_pct_series.max()) if not drawdown_pct_series.empty else 0.0

        # Drawdown duration
        is_in_dd = drawdown_pct_series > 0.001
        dd_durations = []
        cur_dur = 0
        for in_dd in is_in_dd:
            if in_dd:
                cur_dur += 1
            else:
                if cur_dur > 0:
                    dd_durations.append(cur_dur)
                cur_dur = 0
        if cur_dur > 0:
            dd_durations.append(cur_dur)
        max_dd_duration = max(dd_durations) if dd_durations else 0

        calmar_ratio = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Daily returns for Sharpe and Sortino
        daily_returns = eq_series.pct_change().dropna()
        if len(daily_returns) > 5 and daily_returns.std() > 0:
            daily_rf = (1.0 + cls.RISK_FREE_RATE_ANNUAL) ** (1.0 / 252.0) - 1.0
            excess_daily = daily_returns - daily_rf
            sharpe_ratio = float((excess_daily.mean() / daily_returns.std()) * np.sqrt(252.0))

            downside_returns = excess_daily[excess_daily < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 1 else daily_returns.std()
            sortino_ratio = float((excess_daily.mean() / downside_std) * np.sqrt(252.0)) if downside_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Breakdowns
        setup_breakdown = cls._calculate_group_breakdown(trades, "setup_type")
        exit_breakdown = cls._calculate_group_breakdown(trades, "exit_reason")
        sector_breakdown = cls._calculate_group_breakdown(trades, "sector")

        return PerformanceSummary(
            initial_capital_inr=initial_capital,
            final_equity_inr=round(final_equity, 2),
            net_profit_inr=round(net_profit, 2),
            net_return_pct=round(net_return_pct, 2),
            cagr_pct=round(cagr_pct, 2),
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            scratch_trades=scratch_count,
            win_rate_pct=round(win_rate, 2),
            loss_rate_pct=round(loss_rate, 2),
            gross_profit_inr=round(gross_profit, 2),
            gross_loss_inr=round(gross_loss, 2),
            total_friction_inr=round(total_friction, 2),
            profit_factor=round(profit_factor, 2),
            payoff_ratio=round(payoff_ratio, 2),
            avg_trade_pnl_inr=round(net_profit / total_trades, 2),
            avg_win_inr=round(avg_win_inr, 2),
            avg_loss_inr=round(avg_loss_inr, 2),
            avg_win_r=round(avg_win_r, 2),
            avg_loss_r=round(avg_loss_r, 2),
            expectancy_r=round(expectancy_r, 2),
            expectancy_inr=round(expectancy_inr, 2),
            max_drawdown_inr=round(max_dd_inr, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=round(calmar_ratio, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            sortino_ratio=round(sortino_ratio, 2),
            avg_holding_days_all=round(avg_hold_all, 1),
            avg_holding_days_winners=round(avg_hold_win, 1),
            avg_holding_days_losers=round(avg_hold_loss, 1),
            setup_breakdown=setup_breakdown,
            exit_reason_breakdown=exit_breakdown,
            sector_breakdown=sector_breakdown,
        )

    @staticmethod
    def _calculate_group_breakdown(trades: List[BacktestTrade], group_field: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[BacktestTrade]] = {}
        for t in trades:
            val = getattr(t, group_field, "UNKNOWN")
            groups.setdefault(val, []).append(t)

        breakdown = {}
        for name, g_trades in groups.items():
            g_wins = [t for t in g_trades if t.net_pnl_inr > 0]
            g_losses = [t for t in g_trades if t.net_pnl_inr < 0]
            g_profit = sum(t.net_pnl_inr for t in g_trades)
            g_win_rate = (len(g_wins) / len(g_trades)) * 100.0 if g_trades else 0.0
            breakdown[name] = {
                "trades": len(g_trades),
                "wins": len(g_wins),
                "losses": len(g_losses),
                "win_rate_pct": round(g_win_rate, 2),
                "net_pnl_inr": round(g_profit, 2),
                "avg_r": round(sum(t.realized_r_multiple for t in g_trades) / len(g_trades), 2),
            }
        return breakdown
