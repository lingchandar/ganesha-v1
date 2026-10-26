"""Live FYERS candle -> swing analysis pipeline.

The live monitor is intentionally read-only.  It may surface validated setup
candidates, but it must never create a paper/live position by bypassing the
centralized entry authorization path.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd

from src.ingestion.live_ohlcv_buffer import LiveOHLCVBuffer
from src.ingestion.market_data_client import MarketDataClient
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.ingestion.timeframe_aggregator import TimeframeAggregator
from src.risk.active_trade_manager import ActiveTradeManager
from src.setups.swing_setups import SwingSetupSignal
from src.setups.swing_setups import SwingSetupScanner


class LiveAnalysisEngine:
    """Warm up from FYERS history, then surface setup candidates read-only.

    The validated Ganesha swing methodology currently uses ``TREND_PULLBACK``
    as its live-validation setup.  This engine therefore filters out every
    other setup.  Position creation is deliberately not performed here:
    final authorization belongs to the centralized entry gate after all
    required eligibility, prediction, and risk conditions are available.
    """

    DEFAULT_SETUP_TYPE = "TREND_PULLBACK"

    def __init__(
        self,
        min_candles: int = 60,
        max_candles: int = 300,
        analysis_timeframe_minutes: int = 15,
        scanner: Optional[SwingSetupScanner] = None,
        signal_handler: Optional[Callable[[str, list], None]] = None,
        trade_manager: Optional[ActiveTradeManager] = None,
        setup_type_filter: str = DEFAULT_SETUP_TYPE,
    ) -> None:
        if min_candles < 60:
            raise ValueError("min_candles must be >= 60 for the swing scanner")
        if max_candles < min_candles:
            raise ValueError("max_candles must be >= min_candles")
        if setup_type_filter != self.DEFAULT_SETUP_TYPE:
            raise ValueError(
                "Live swing validation is locked to TREND_PULLBACK until another setup "
                "passes the same historical validation standard."
            )

        self.min_candles = min_candles
        self.analysis_timeframe_minutes = analysis_timeframe_minutes
        self.setup_type_filter = setup_type_filter
        self.source_buffer = LiveOHLCVBuffer(
            max_candles=max_candles * analysis_timeframe_minutes
        )
        self.aggregator = TimeframeAggregator(analysis_timeframe_minutes)
        self.analysis_buffers = LiveOHLCVBuffer(max_candles=max_candles)
        self.scanner = scanner or SwingSetupScanner()
        self.signal_handler = signal_handler
        self.trade_manager = trade_manager or ActiveTradeManager()

    def warm_up(self, symbol: str, client: MarketDataClient, lookback_days: int = 7) -> int:
        """Load completed analysis candles from FYERS before live monitoring starts."""
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")

        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=lookback_days)

        history = client.fetch_historical_candles(
            symbol,
            start,
            end,
            resolution=str(self.analysis_timeframe_minutes),
        )
        if history.empty:
            return 0

        for row in history.tail(self.analysis_buffers.max_candles).itertuples(index=False):
            candle = OneMinuteCandle(
                symbol=symbol,
                timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                open_price=float(row.open_price),
                high_price=float(row.high_price),
                low_price=float(row.low_price),
                close_price=float(row.close_price),
                volume_traded=int(row.volume_traded),
                tick_count=self.analysis_timeframe_minutes,
            )
            self.analysis_buffers.add(candle)

        return self.analysis_buffers.count(symbol)

    def on_candle(self, candle: OneMinuteCandle) -> list:
        """Accept a completed 1-minute candle and surface Trend Pullback candidates.

        No position is opened here.  In particular, this method must not turn a
        setup signal into an entry merely because sizing or circuit-breaker
        checks happen to pass; the centralized entry gate is the only valid
        authorization boundary.
        """
        self.source_buffer.add(candle)
        source_df = self.source_buffer.get(candle.symbol)
        analysis_df = self.aggregator.aggregate(source_df)

        if analysis_df.empty:
            return []

        latest = analysis_df.iloc[-1]
        analysis_candle = OneMinuteCandle(
            symbol=candle.symbol,
            timestamp=latest["timestamp"].to_pydatetime(),
            open_price=float(latest["open_price"]),
            high_price=float(latest["high_price"]),
            low_price=float(latest["low_price"]),
            close_price=float(latest["close_price"]),
            volume_traded=int(latest["volume_traded"]),
            tick_count=int(latest["source_candle_count"]),
        )

        existing = self.analysis_buffers.get(candle.symbol)
        if not existing.empty and existing.iloc[-1]["timestamp"] >= latest["timestamp"]:
            return []

        self.analysis_buffers.add(analysis_candle)

        if self.analysis_buffers.count(candle.symbol) < self.min_candles:
            return []

        df = self.analysis_buffers.get(candle.symbol)
        signals = self.scanner.scan_all_setups(
            ticker_symbol=candle.symbol,
            df=df,
        )
        signals = [
            signal
            for signal in signals
            if isinstance(signal, SwingSetupSignal)
            and signal.setup_type == self.setup_type_filter
        ]

        if self.signal_handler:
            self.signal_handler(candle.symbol, signals)
        return signals

    def dataframe(self, symbol: str) -> pd.DataFrame:
        """Return completed analysis-timeframe candles for a symbol."""
        return self.analysis_buffers.get(symbol)
