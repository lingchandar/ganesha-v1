"""Live FYERS tick -> daily swing-analysis pipeline.

The validated swing methodology is daily-bar based. Live FYERS ticks are used
only to build the current session's daily OHLCV candle; setup evaluation happens
once the session is complete. No position is created by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingestion.live_ohlcv_buffer import LiveOHLCVBuffer
from src.ingestion.market_data_client import MarketDataClient
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.risk.active_trade_manager import ActiveTradeManager
from src.setups.swing_setups import SwingSetupScanner, SwingSetupSignal


IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE = time(15, 29)


@dataclass
class _WorkingDailyCandle:
    symbol: str
    session_date: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume_traded: int = 0
    minute_count: int = 0

    def update(self, candle: OneMinuteCandle) -> None:
        self.high_price = max(self.high_price, float(candle.high_price))
        self.low_price = min(self.low_price, float(candle.low_price))
        self.close_price = float(candle.close_price)
        self.volume_traded += int(candle.volume_traded)
        self.minute_count += 1

    def as_candle(self) -> OneMinuteCandle:
        return OneMinuteCandle(
            symbol=self.symbol,
            timestamp=datetime.combine(self.session_date, time.min, tzinfo=timezone.utc),
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            close_price=self.close_price,
            volume_traded=self.volume_traded,
            tick_count=self.minute_count,
        )


class LiveAnalysisEngine:
    """Warm up from daily FYERS history, then validate Trend Pullback live."""

    DEFAULT_SETUP_TYPE = "TREND_PULLBACK"
    DEFAULT_WARMUP_CALENDAR_DAYS = 450

    def __init__(
        self,
        min_candles: int = 300,
        max_candles: int = 500,
        analysis_timeframe_minutes: int = 1,
        scanner: Optional[SwingSetupScanner] = None,
        signal_handler: Optional[Callable[[str, list], None]] = None,
        trade_manager: Optional[ActiveTradeManager] = None,
        setup_type_filter: str = DEFAULT_SETUP_TYPE,
    ) -> None:
        if min_candles < 300:
            raise ValueError("min_candles must be >= 300 for the validated daily swing methodology")
        if max_candles < min_candles:
            raise ValueError("max_candles must be >= min_candles")
        if analysis_timeframe_minutes != 1:
            raise ValueError("Live swing analysis consumes completed 1-minute candles and evaluates daily bars")
        if setup_type_filter != self.DEFAULT_SETUP_TYPE:
            raise ValueError(
                "Live swing validation is locked to TREND_PULLBACK until another setup "
                "passes the same historical validation standard."
            )

        self.min_candles = min_candles
        self.max_candles = max_candles
        self.setup_type_filter = setup_type_filter
        self.analysis_buffers = LiveOHLCVBuffer(max_candles=max_candles)
        self.scanner = scanner or SwingSetupScanner()
        self.signal_handler = signal_handler
        self.trade_manager = trade_manager or ActiveTradeManager()
        self._working_days: dict[str, _WorkingDailyCandle] = {}
        self._finalized_sessions: dict[str, date] = {}

    @staticmethod
    def _session_date(timestamp: datetime) -> date:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(IST).date()

    @staticmethod
    def _local_time(timestamp: datetime) -> time:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(IST).time().replace(second=0, microsecond=0)

    def warm_up(self, symbol: str, client: MarketDataClient, lookback_days: int = DEFAULT_WARMUP_CALENDAR_DAYS) -> int:
        """Load enough completed daily history for EMA-200 and the 300-bar warm-up."""
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")

        calendar_days = max(int(lookback_days), self.DEFAULT_WARMUP_CALENDAR_DAYS)
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=calendar_days)

        history = client.fetch_historical_candles(
            symbol,
            start,
            end,
            resolution="D",
        )
        if history.empty:
            return 0

        for row in history.tail(self.analysis_buffers.max_candles).itertuples(index=False):
            timestamp = pd.Timestamp(row.timestamp).to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            candle = OneMinuteCandle(
                symbol=symbol,
                timestamp=timestamp,
                open_price=float(row.open_price),
                high_price=float(row.high_price),
                low_price=float(row.low_price),
                close_price=float(row.close_price),
                volume_traded=int(row.volume_traded),
                tick_count=1,
            )
            self.analysis_buffers.add(candle)

        return self.analysis_buffers.count(symbol)

    def _finalize_session(self, symbol: str, session_date: date) -> list:
        working = self._working_days.get(symbol)
        if working is None or working.session_date != session_date:
            return []
        if self._finalized_sessions.get(symbol) == session_date:
            return []

        self._finalized_sessions[symbol] = session_date
        self.analysis_buffers.add(working.as_candle())

        if self.analysis_buffers.count(symbol) < self.min_candles:
            return []

        df = self.analysis_buffers.get(symbol)
        signals = self.scanner.scan_all_setups(ticker_symbol=symbol, df=df)
        signals = [
            signal
            for signal in signals
            if isinstance(signal, SwingSetupSignal)
            and signal.setup_type == self.setup_type_filter
        ]

        if self.signal_handler:
            self.signal_handler(symbol, signals)
        return signals

    def on_candle(self, candle: OneMinuteCandle) -> list:
        """Consume a completed 1-minute candle and scan only completed daily sessions."""
        symbol = candle.symbol
        session_date = self._session_date(candle.timestamp)
        local_time = self._local_time(candle.timestamp)
        working = self._working_days.get(symbol)

        if working is not None and session_date < working.session_date:
            return []

        if working is not None and session_date > working.session_date:
            previous = self._finalize_session(symbol, working.session_date)
            working = None
        else:
            previous = []

        if self._finalized_sessions.get(symbol) == session_date:
            return []

        if working is None:
            working = _WorkingDailyCandle(
                symbol=symbol,
                session_date=session_date,
                open_price=float(candle.open_price),
                high_price=float(candle.high_price),
                low_price=float(candle.low_price),
                close_price=float(candle.close_price),
                volume_traded=int(candle.volume_traded),
                minute_count=1,
            )
            self._working_days[symbol] = working
        else:
            working.update(candle)

        if local_time >= MARKET_CLOSE:
            return self._finalize_session(symbol, session_date)
        return previous

    def finalize_session(self, symbol: str) -> list:
        """Explicitly finalize the current session, useful for controlled shutdown/tests."""
        working = self._working_days.get(symbol)
        if working is None:
            return []
        return self._finalize_session(symbol, working.session_date)

    def dataframe(self, symbol: str) -> pd.DataFrame:
        """Return completed daily candles used by the swing scanner."""
        return self.analysis_buffers.get(symbol)
