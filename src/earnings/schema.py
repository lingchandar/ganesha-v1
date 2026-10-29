"""Point-in-time earnings event schema for Ganesha V2.

The schema stores facts known at event time. It does not manufacture beats,
misses, or market reactions when source data is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class EarningsState(str, Enum):
    APPROACHING = "EARNINGS_APPROACHING"
    IMMINENT = "EARNINGS_IMMINENT"
    RELEASED = "EARNINGS_RELEASED"
    BEAT = "EARNINGS_BEAT"
    MISS = "EARNINGS_MISS"
    MIXED = "EARNINGS_MIXED"
    POST_CONFIRMATION = "POST_EARNINGS_CONFIRMATION"
    POST_FAILURE = "POST_EARNINGS_FAILURE"
    UNKNOWN = "EARNINGS_UNKNOWN"


@dataclass(frozen=True)
class EarningsEvent:
    ticker_symbol: str
    event_date: date
    state: EarningsState = EarningsState.UNKNOWN
    expected_revenue: Optional[float] = None
    actual_revenue: Optional[float] = None
    expected_eps: Optional[float] = None
    actual_eps: Optional[float] = None
    expected_profit: Optional[float] = None
    actual_profit: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    margin_pct: Optional[float] = None
    guidance_score: Optional[float] = None
    cashflow_score: Optional[float] = None
    price_reaction_pct: Optional[float] = None
    volume_ratio: Optional[float] = None
    sector_reaction_pct: Optional[float] = None
    source: str = "unknown"

    @property
    def surprise_available(self) -> bool:
        return any(value is not None for value in (self.expected_revenue, self.actual_revenue, self.expected_eps, self.actual_eps, self.expected_profit, self.actual_profit))

    def beat_score(self) -> Optional[float]:
        values: list[float] = []
        for expected, actual in ((self.expected_revenue, self.actual_revenue), (self.expected_eps, self.actual_eps), (self.expected_profit, self.actual_profit)):
            if expected is not None and actual is not None and expected != 0:
                values.append((actual - expected) / abs(expected))
        return sum(values) / len(values) if values else None
