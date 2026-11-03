"""Adapter boundary for structured earnings/news context.

An external LLM/provider can implement this interface later. The strategy layer
must consume structured context and never a free-form BUY/SELL instruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.earnings.schema import EarningsEvent


@dataclass(frozen=True)
class EarningsContext:
    event: EarningsEvent | None
    summary: str
    confidence: float
    reason_codes: tuple[str, ...]


class EarningsContextProvider(Protocol):
    def build_context(self, ticker_symbol: str, events: Sequence[EarningsEvent]) -> EarningsContext: ...


class DeterministicEarningsContextProvider:
    """Safe default provider; no LLM and no trading decision."""

    def build_context(self, ticker_symbol: str, events: Sequence[EarningsEvent]) -> EarningsContext:
        relevant = sorted((event for event in events if event.ticker_symbol == ticker_symbol), key=lambda event: event.event_date)
        if not relevant:
            return EarningsContext(None, "No structured earnings event available.", 0.0, ("NO_EARNINGS_DATA",))
        event = relevant[-1]
        if not event.surprise_available:
            return EarningsContext(event, "Earnings event exists but surprise fields are unavailable.", 0.25, ("EARNINGS_DATA_INCOMPLETE",))
        beat = event.beat_score()
        if beat is None:
            return EarningsContext(event, "Earnings event exists without a computable surprise score.", 0.25, ("EARNINGS_SURPRISE_UNAVAILABLE",))
        if beat > 0.02:
            return EarningsContext(event, "Structured earnings surprise is positive; price reaction must still confirm.", 0.75, ("EARNINGS_POSITIVE_SURPRISE",))
        if beat < -0.02:
            return EarningsContext(event, "Structured earnings surprise is negative; price reaction must still be evaluated.", 0.75, ("EARNINGS_NEGATIVE_SURPRISE",))
        return EarningsContext(event, "Structured earnings surprise is mixed/near neutral.", 0.60, ("EARNINGS_MIXED_SURPRISE",))
