"""Market Pulse domain model for Ganesha v1.

This module is deliberately data-source agnostic. It defines the canonical
market-session state and the snapshot contract that live FYERS, FII/DII, MF,
sector and dashboard adapters will populate later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class MarketSession(str, Enum):
    PRE_OPEN = "PRE_OPEN"
    OPEN = "OPEN"
    INTRADAY = "INTRADAY"
    POST_CLOSE = "POST_CLOSE"
    CLOSED = "CLOSED"


def market_session(at: Optional[datetime] = None) -> MarketSession:
    """Return NSE equity session state for an IST timestamp.

    This is only the regular-session clock. Exchange holidays, special
    sessions and settlement-specific windows belong to a future calendar
    adapter and must not be guessed here.
    """
    current = (at or datetime.now(IST)).astimezone(IST)
    if current.weekday() >= 5:
        return MarketSession.CLOSED

    t = current.time()
    if time(9, 0) <= t < time(9, 15):
        return MarketSession.PRE_OPEN
    if time(9, 15) <= t < time(15, 30):
        return MarketSession.INTRADAY
    if time(15, 30) <= t < time(16, 0):
        return MarketSession.POST_CLOSE
    return MarketSession.CLOSED


@dataclass(frozen=True)
class MarketStockSnapshot:
    ticker_symbol: str
    price: float
    change_pct: float = 0.0
    volume: float = 0.0
    volume_ratio_20d: float = 0.0
    relative_strength_20d: float = 0.0
    sector: str = "UNKNOWN"
    momentum_score: float = 0.0
    money_flow_score: float = 0.0
    momentum_event: bool = False


@dataclass(frozen=True)
class MoneyFlowSnapshot:
    fii_net: Optional[float] = None
    dii_net: Optional[float] = None
    source_date: Optional[str] = None


@dataclass(frozen=True)
class MutualFundSnapshot:
    equity_net_flow: Optional[float] = None
    debt_net_flow: Optional[float] = None
    gold_net_flow: Optional[float] = None
    source_date: Optional[str] = None
    is_intraday: bool = False


@dataclass(frozen=True)
class MarketPulse:
    as_of: datetime
    session: MarketSession
    stocks: tuple[MarketStockSnapshot, ...] = field(default_factory=tuple)
    money_flow: MoneyFlowSnapshot = field(default_factory=MoneyFlowSnapshot)
    mutual_funds: MutualFundSnapshot = field(default_factory=MutualFundSnapshot)

    @property
    def top_momentum(self) -> tuple[MarketStockSnapshot, ...]:
        return tuple(sorted(self.stocks, key=lambda x: x.momentum_score, reverse=True)[:10])

    @property
    def unusual_volume(self) -> tuple[MarketStockSnapshot, ...]:
        return tuple(
            sorted(self.stocks, key=lambda x: x.volume_ratio_20d, reverse=True)[:10]
        )

    @property
    def positive_money_flow(self) -> tuple[MarketStockSnapshot, ...]:
        return tuple(
            sorted(self.stocks, key=lambda x: x.money_flow_score, reverse=True)[:10]
        )


def build_market_pulse(
    stocks: Iterable[MarketStockSnapshot],
    *,
    as_of: Optional[datetime] = None,
    money_flow: Optional[MoneyFlowSnapshot] = None,
    mutual_funds: Optional[MutualFundSnapshot] = None,
) -> MarketPulse:
    """Build the canonical pulse object used by scanners and dashboards."""
    timestamp = (as_of or datetime.now(IST)).astimezone(IST)
    return MarketPulse(
        as_of=timestamp,
        session=market_session(timestamp),
        stocks=tuple(stocks),
        money_flow=money_flow or MoneyFlowSnapshot(),
        mutual_funds=mutual_funds or MutualFundSnapshot(),
    )
