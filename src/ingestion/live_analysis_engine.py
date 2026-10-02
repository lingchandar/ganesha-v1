"""Live FYERS candle -> swing analysis pipeline."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd

from src.ingestion.live_ohlcv_buffer import LiveOHLCVBuffer
from src.ingestion.market_data_client import MarketDataClient
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.ingestion.timeframe_aggregator import TimeframeAggregator
from src.risk.active_trade_manager import ActiveTradeManager
from src.setups.swing_setups import SwingSetupScanner


class LiveAnalysisEngine:
    """Warm up from FYERS history, then continue from completed live candles."""

    def __init__(
        self,
        min_candles: int = 60,
        max_candles: int = 300,
        analysis_timeframe_minutes: int = 15,
        scanner: Optional[SwingSetupScanner] = None,
        signal_handler: Optional[Callable[[str, list], None]] = None,
        trade_manager: Optional[ActiveTradeManager] = None,
    ) -> None:
        if min_candles < 60:
            raise ValueError("min_candles must be >= 60 for the swing scanner")
        if max_candles < min_candles:
            raise ValueError("max_candles must be >= min_candles")

        self.min_candles = min_candles
        self.analysis_timeframe_minutes = analysis_timeframe_minutes
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
        """Accept a completed 1-minute candle and scan on completed analysis bars."""
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

        # Register the first qualifying signal as a paper trade. A symbol can
        # have only one active position; repeated setup detections are ignored
        # by ActiveTradeManager rather than creating duplicate positions.
        if signals:
            for signal in signals:
                if self.trade_manager.has_active_trade(candle.symbol):
                    break
                try:
                    self.trade_manager.open_trade(
                        signal=signal,
                        entry_date=latest["timestamp"].date(),
                    )
                except ValueError:
                    # Invalid/duplicate setup must never break the live feed.
                    continue

        if self.signal_handler:
            self.signal_handler(candle.symbol, signals)
        return signals

    def dataframe(self, symbol: str) -> pd.DataFrame:
        """Return completed analysis-timeframe candles for a symbol."""
        return self.analysis_buffers.get(symbol)
