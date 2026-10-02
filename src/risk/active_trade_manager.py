"""Stateful manager for active Ganesha swing trades.

This module bridges entry signals and the existing daily-bar PrecisionExitEngine.
It is deliberately read-only: it tracks decisions and never places orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from src.risk.exit_engine import ExitEvaluationResult, PrecisionExitEngine
from src.setups.swing_setups import SwingSetupSignal


@dataclass
class ClosedTrade:
    ticker_symbol: str
    setup_type: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    stop_loss: float
    target_price: float
    shares: int
    holding_days: int
    exit_reason: str
    realized_r_multiple: float
    pnl_percentage: float
    pnl_inr: float


@dataclass
class ActiveTrade:
    """Runtime state for one open swing position."""

    ticker_symbol: str
    setup_type: str
    entry_date: date
    entry_price: float
    stop_loss: float
    target_price: float
    invalidation_level: float
    shares: int = 0
    days_held: int = 0
    last_evaluated_date: Optional[date] = None
    is_open: bool = True

    @classmethod
    def from_signal(
        cls,
        signal: SwingSetupSignal,
        entry_date: date,
        shares: int = 0,
    ) -> "ActiveTrade":
        return cls(
            ticker_symbol=signal.ticker_symbol,
            setup_type=signal.setup_type,
            entry_date=entry_date,
            entry_price=float(signal.entry_price),
            stop_loss=float(signal.stop_loss),
            target_price=float(signal.target_price),
            invalidation_level=float(signal.invalidation_level),
            shares=int(shares),
            last_evaluated_date=entry_date,
        )


class ActiveTradeManager:
    """Track open trades and evaluate their daily exit decisions."""

    def __init__(self, exit_engine=PrecisionExitEngine) -> None:
        self.exit_engine = exit_engine
        self._active_trades: Dict[str, ActiveTrade] = {}
        self._closed_trades: Dict[str, ExitEvaluationResult] = {}
        self._closed_trade_records: List[ClosedTrade] = []

    @property
    def active_trades(self) -> Dict[str, ActiveTrade]:
        return dict(self._active_trades)

    @property
    def closed_trades(self) -> Dict[str, ExitEvaluationResult]:
        return dict(self._closed_trades)

    @property
    def closed_trade_records(self) -> List[ClosedTrade]:
        return list(self._closed_trade_records)

    def has_active_trade(self, ticker_symbol: str) -> bool:
        return ticker_symbol in self._active_trades

    def open_trade(
        self,
        signal: SwingSetupSignal,
        entry_date: date,
        shares: int = 0,
    ) -> ActiveTrade:
        """Register an entry signal as an active trade.

        A symbol can have only one active position in v1.
        """
        symbol = signal.ticker_symbol
        if self.has_active_trade(symbol):
            raise ValueError(f"ACTIVE_TRADE_EXISTS: {symbol}")

        if signal.entry_price <= signal.stop_loss:
            raise ValueError(
                f"INVALID_TRADE_RISK: entry {signal.entry_price} "
                f"must exceed stop {signal.stop_loss}"
            )

        trade = ActiveTrade.from_signal(signal, entry_date, shares)
        self._active_trades[symbol] = trade
        return trade

    def evaluate_daily_candle(
        self,
        ticker_symbol: str,
        evaluation_date: date,
        daily_candle: Dict[str, float],
        has_upcoming_earnings: bool = False,
    ) -> ExitEvaluationResult:
        """Evaluate one completed daily candle for an open trade."""
        trade = self._active_trades.get(ticker_symbol)
        if trade is None:
            raise KeyError(f"NO_ACTIVE_TRADE: {ticker_symbol}")

        if evaluation_date < trade.entry_date:
            raise ValueError("evaluation_date cannot be before entry_date")

        if (
            trade.last_evaluated_date is None
            or evaluation_date > trade.last_evaluated_date
        ):
            if evaluation_date > trade.entry_date:
                trade.days_held += 1
            trade.last_evaluated_date = evaluation_date

        result = self.exit_engine.evaluate_active_trade(
            entry_price=trade.entry_price,
            stop_loss=trade.stop_loss,
            target_price=trade.target_price,
            invalidation_level=trade.invalidation_level,
            days_held=trade.days_held,
            daily_candle=daily_candle,
            has_upcoming_earnings=has_upcoming_earnings,
        )

        if result.is_exit_triggered:
            trade.is_open = False
            self._closed_trades[ticker_symbol] = result
            self._closed_trade_records.append(ClosedTrade(ticker_symbol=trade.ticker_symbol, setup_type=trade.setup_type, entry_date=trade.entry_date, exit_date=evaluation_date, entry_price=trade.entry_price, exit_price=result.exit_price, stop_loss=trade.stop_loss, target_price=trade.target_price, shares=trade.shares, holding_days=result.holding_days_elapsed, exit_reason=result.exit_reason, realized_r_multiple=result.realized_r_multiple, pnl_percentage=result.pnl_percentage, pnl_inr=round((result.exit_price - trade.entry_price) * trade.shares, 2)))
            del self._active_trades[ticker_symbol]

        return result

    def clear_closed_history(self) -> None:
        self._closed_trades.clear()
