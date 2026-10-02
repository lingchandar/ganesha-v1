"""Stateful manager for active Ganesha swing trades.

This module bridges entry signals and the existing daily-bar PrecisionExitEngine.
It is deliberately read-only: it tracks decisions and never places orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

from src.risk.exit_engine import ExitEvaluationResult, PrecisionExitEngine
from src.setups.swing_setups import SwingSetupSignal


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

    @property
    def active_trades(self) -> Dict[str, ActiveTrade]:
        return dict(self._active_trades)

    @property
    def closed_trades(self) -> Dict[str, ExitEvaluationResult]:
        return dict(self._closed_trades)

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
            del self._active_trades[ticker_symbol]

        return result

    def clear_closed_history(self) -> None:
        self._closed_trades.clear()
