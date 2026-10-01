"""Build deterministic 1-minute OHLCV candles from FYERS live ticks.

FYERS SymbolUpdate volume is cumulative traded volume for the session, so the
builder converts volume changes into per-minute volume instead of summing the
cumulative value repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Dict, Optional

from src.ingestion.fyers_realtime import LiveMarketTick


@dataclass(frozen=True)
class OneMinuteCandle:
    """Completed one-minute candle built from live ticks."""

    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume_traded: int
    tick_count: int


@dataclass
class _WorkingCandle:
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume_traded: int = 0
    tick_count: int = 0


class OneMinuteCandleBuilder:
    """Thread-safe, deterministic live tick -> 1-minute candle aggregator."""

    def __init__(
        self,
        candle_handler: Optional[Callable[[OneMinuteCandle], None]] = None,
    ) -> None:
        self._candle_handler = candle_handler
        self._working: Dict[str, _WorkingCandle] = {}
        self._last_cumulative_volume: Dict[str, int] = {}
        self._lock = RLock()

    @staticmethod
    def _minute_start(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).replace(
            second=0,
            microsecond=0,
        )

    def _emit(self, candle: OneMinuteCandle) -> None:
        if self._candle_handler:
            self._candle_handler(candle)

    def update(self, tick: LiveMarketTick) -> Optional[OneMinuteCandle]:
        """Add one tick and emit/return the previous candle when its minute ends."""
        if tick.ltp is None or tick.timestamp is None:
            return None

        minute = self._minute_start(tick.timestamp)
        ltp = float(tick.ltp)

        with self._lock:
            previous_cumulative = self._last_cumulative_volume.get(tick.symbol)
            cumulative = tick.volume_traded
            volume_delta = 0

            if cumulative is not None:
                cumulative = max(0, int(cumulative))
                if previous_cumulative is not None:
                    volume_delta = max(0, cumulative - previous_cumulative)
                self._last_cumulative_volume[tick.symbol] = cumulative

            current = self._working.get(tick.symbol)

            if current is None:
                self._working[tick.symbol] = _WorkingCandle(
                    symbol=tick.symbol,
                    timestamp=minute,
                    open_price=ltp,
                    high_price=ltp,
                    low_price=ltp,
                    close_price=ltp,
                    volume_traded=volume_delta,
                    tick_count=1,
                )
                return None

            if minute < current.timestamp:
                # Ignore late/out-of-order ticks rather than corrupting history.
                return None

            if minute == current.timestamp:
                current.high_price = max(current.high_price, ltp)
                current.low_price = min(current.low_price, ltp)
                current.close_price = ltp
                current.volume_traded += volume_delta
                current.tick_count += 1
                return None

            completed = OneMinuteCandle(
                symbol=current.symbol,
                timestamp=current.timestamp,
                open_price=current.open_price,
                high_price=current.high_price,
                low_price=current.low_price,
                close_price=current.close_price,
                volume_traded=current.volume_traded,
                tick_count=current.tick_count,
            )

            self._working[tick.symbol] = _WorkingCandle(
                symbol=tick.symbol,
                timestamp=minute,
                open_price=ltp,
                high_price=ltp,
                low_price=ltp,
                close_price=ltp,
                volume_traded=volume_delta,
                tick_count=1,
            )

        self._emit(completed)
        return completed

    def flush(self, symbol: Optional[str] = None) -> list[OneMinuteCandle]:
        """Finalize currently open candles, useful at shutdown/session boundaries."""
        with self._lock:
            symbols = [symbol] if symbol else list(self._working)
            completed: list[OneMinuteCandle] = []

            for ticker in symbols:
                current = self._working.pop(ticker, None)
                if current is None:
                    continue
                completed.append(
                    OneMinuteCandle(
                        symbol=current.symbol,
                        timestamp=current.timestamp,
                        open_price=current.open_price,
                        high_price=current.high_price,
                        low_price=current.low_price,
                        close_price=current.close_price,
                        volume_traded=current.volume_traded,
                        tick_count=current.tick_count,
                    )
                )

        for candle in completed:
            self._emit(candle)
        return completed
