"""Read-only 59-stock live scanner for Ganesha v1.

The scanner reuses the existing FYERS WebSocket, one-minute candle builder and
MarketPulse contracts. It does not place orders and it does not fabricate FII,
DII or mutual-fund data.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, time
from statistics import mean
from threading import RLock
from typing import Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

from src.ingestion.fyers_realtime import FyersRealtimeClient, LiveMarketTick
from src.ingestion.live_ohlcv_buffer import LiveOHLCVBuffer
from src.ingestion.one_minute_candle_builder import OneMinuteCandleBuilder
from src.market_pulse.engine import MarketPulse, MarketStockSnapshot, build_market_pulse

IST = ZoneInfo("Asia/Kolkata")

# Controlled v1 universe. These are FYERS/NSE symbols, not arbitrary discovery.
NSE_59_SYMBOLS: tuple[str, ...] = (
    "NSE:ADANIENT-EQ", "NSE:ADANIPORTS-EQ", "NSE:APOLLOHOSP-EQ",
    "NSE:ASIANPAINT-EQ", "NSE:AUBANK-EQ", "NSE:AXISBANK-EQ",
    "NSE:BAJAJ-AUTO-EQ", "NSE:BAJAJFINSV-EQ", "NSE:BAJFINANCE-EQ",
    "NSE:BANKBARODA-EQ", "NSE:BEL-EQ", "NSE:BHARTIARTL-EQ",
    "NSE:BSE-EQ", "NSE:CANBK-EQ", "NSE:CIPLA-EQ", "NSE:COALINDIA-EQ",
    "NSE:DRREDDY-EQ", "NSE:EICHERMOT-EQ", "NSE:ETERNAL-EQ",
    "NSE:FEDERALBNK-EQ", "NSE:GRASIM-EQ", "NSE:HCLTECH-EQ",
    "NSE:HDFCBANK-EQ", "NSE:HDFCLIFE-EQ", "NSE:HINDALCO-EQ",
    "NSE:HINDUNILVR-EQ", "NSE:ICICIBANK-EQ", "NSE:IDFCFIRSTB-EQ",
    "NSE:INDIGO-EQ", "NSE:INDUSINDBK-EQ", "NSE:INFY-EQ", "NSE:ITC-EQ",
    "NSE:JIOFIN-EQ", "NSE:JSWSTEEL-EQ", "NSE:KOTAKBANK-EQ", "NSE:LT-EQ",
    "NSE:M&M-EQ", "NSE:MARUTI-EQ", "NSE:MAXHEALTH-EQ", "NSE:NESTLEIND-EQ",
    "NSE:NTPC-EQ", "NSE:ONGC-EQ", "NSE:PNB-EQ", "NSE:POWERGRID-EQ",
    "NSE:RELIANCE-EQ", "NSE:SBILIFE-EQ", "NSE:SBIN-EQ", "NSE:SHRIRAMFIN-EQ",
    "NSE:SUNPHARMA-EQ", "NSE:TATACONSUM-EQ", "NSE:TATASTEEL-EQ", "NSE:TCS-EQ",
    "NSE:TECHM-EQ", "NSE:TITAN-EQ", "NSE:TMPV-EQ", "NSE:TRENT-EQ",
    "NSE:ULTRACEMCO-EQ", "NSE:UNIONBANK-EQ", "NSE:YESBANK-EQ",
)


@dataclass(frozen=True)
class LiveScannerSnapshot:
    """Current transparent scanner state for one stock."""

    ticker_symbol: str
    price: float
    change_pct: float
    return_3m_pct: float
    return_5m_pct: float
    session_volume_ratio: float
    range_ratio: float
    momentum_score: float
    money_flow_score: float
    momentum_event: bool
    candle_count: int


class LiveMarketScanner:
    """Build a live MarketPulse from the existing FYERS tick stream."""

    SESSION_OPEN = time(9, 15)
    SESSION_CLOSE = time(15, 30)
    MAX_CANDLES = 120

    def __init__(
        self,
        *,
        symbols: Iterable[str] = NSE_59_SYMBOLS,
        baseline_daily_volume: Optional[Mapping[str, float]] = None,
        baseline_daily_volume_history: Optional[Mapping[str, Iterable[float]]] = None,
        tick_client: Optional[FyersRealtimeClient] = None,
    ) -> None:
        requested = tuple(sorted({str(s).strip() for s in symbols if str(s).strip()}))
        if not requested:
            raise ValueError("At least one scanner symbol is required")
        unknown = set(requested) - set(NSE_59_SYMBOLS)
        if unknown:
            raise ValueError(f"Symbols outside the controlled v1 universe: {sorted(unknown)}")

        self.symbols = requested
        self._baseline_daily_volume = dict(baseline_daily_volume or {})
        if baseline_daily_volume_history:
            for symbol, values in baseline_daily_volume_history.items():
                clean = [float(v) for v in values if float(v) > 0]
                if clean:
                    self._baseline_daily_volume[symbol] = mean(clean[-20:])

        self.buffer = LiveOHLCVBuffer(max_candles=self.MAX_CANDLES)
        self.candle_builder = OneMinuteCandleBuilder(candle_handler=self._on_candle)
        self._snapshots: dict[str, LiveScannerSnapshot] = {}
        self._last_ticks: dict[str, LiveMarketTick] = {}
        self._candles: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_CANDLES))
        self._lock = RLock()
        self.tick_client = tick_client or FyersRealtimeClient(tick_handler=self.process_tick)

    def process_tick(self, tick: LiveMarketTick) -> None:
        """Process one normalized FYERS tick; completed candles update the pulse."""
        if tick.symbol not in self.symbols:
            return
        with self._lock:
            self._last_ticks[tick.symbol] = tick
        self.candle_builder.update(tick)

    def _on_candle(self, candle) -> None:
        if candle.symbol not in self.symbols:
            return
        with self._lock:
            candles = self._candles[candle.symbol]
            candles.append(candle)
            self.buffer.add(candle)
            tick = self._last_ticks.get(candle.symbol)
            self._snapshots[candle.symbol] = self._calculate_snapshot(candle.symbol, tick)

    def _calculate_snapshot(self, symbol: str, tick: Optional[LiveMarketTick]) -> LiveScannerSnapshot:
        candles = list(self._candles[symbol])
        last = candles[-1]
        previous_close = tick.previous_close if tick and tick.previous_close else last.open_price
        change_pct = ((last.close_price / previous_close) - 1.0) * 100.0 if previous_close else 0.0

        def candle_return(count: int) -> float:
            if len(candles) <= count:
                return 0.0
            base = candles[-count - 1].close_price
            return ((last.close_price / base) - 1.0) * 100.0 if base else 0.0

        prior_ranges = [c.high_price - c.low_price for c in candles[-21:-1]]
        avg_range = mean(prior_ranges) if prior_ranges else 0.0
        current_range = last.high_price - last.low_price
        range_ratio = current_range / avg_range if avg_range > 0 else 0.0

        volume_ratio = self._session_volume_ratio(symbol, tick)
        return_3m = candle_return(3)
        return_5m = candle_return(5)

        # Transparent heuristic ranking only. ML is intentionally not wired here.
        momentum_score = (
            max(-5.0, min(5.0, change_pct)) * 8.0
            + max(-5.0, min(5.0, return_3m)) * 5.0
            + max(-5.0, min(5.0, return_5m)) * 3.0
            + max(0.0, min(4.0, volume_ratio - 1.0)) * 8.0
            + max(0.0, min(3.0, range_ratio - 1.0)) * 5.0
        )
        money_flow_score = max(-5.0, min(5.0, change_pct)) * max(0.0, volume_ratio)
        momentum_event = (
            change_pct > 0.35
            and return_3m > 0.50
            and volume_ratio >= 1.75
            and range_ratio >= 1.25
        )

        return LiveScannerSnapshot(
            ticker_symbol=symbol,
            price=last.close_price,
            change_pct=change_pct,
            return_3m_pct=return_3m,
            return_5m_pct=return_5m,
            session_volume_ratio=volume_ratio,
            range_ratio=range_ratio,
            momentum_score=momentum_score,
            money_flow_score=money_flow_score,
            momentum_event=momentum_event,
            candle_count=len(candles),
        )

    def _session_volume_ratio(self, symbol: str, tick: Optional[LiveMarketTick]) -> float:
        """Compare current cumulative volume with expected volume by session time."""
        if not tick or tick.volume_traded is None:
            return 0.0
        baseline = self._baseline_daily_volume.get(symbol)
        if not baseline or baseline <= 0 or not tick.timestamp:
            return 0.0
        local = tick.timestamp.astimezone(IST)
        if local.time() <= self.SESSION_OPEN:
            return 0.0
        elapsed = (local.hour * 60 + local.minute) - (9 * 60 + 15)
        total_minutes = 375.0
        expected = baseline * min(1.0, max(0.0, elapsed / total_minutes))
        return float(tick.volume_traded) / expected if expected > 0 else 0.0

    def snapshots(self) -> tuple[LiveScannerSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots.values())

    def market_pulse(self, *, as_of: Optional[datetime] = None) -> MarketPulse:
        """Build the canonical MarketPulse from stocks with completed candles."""
        with self._lock:
            snapshots = tuple(self._snapshots.values())
        stocks = [
            MarketStockSnapshot(
                ticker_symbol=s.ticker_symbol,
                price=s.price,
                change_pct=s.change_pct,
                volume=(self._last_ticks.get(s.ticker_symbol).volume_traded or 0)
                if self._last_ticks.get(s.ticker_symbol) else 0,
                volume_ratio_20d=s.session_volume_ratio,
                momentum_score=s.momentum_score,
                money_flow_score=s.money_flow_score,
                momentum_event=s.momentum_event,
            )
            for s in snapshots
        ]
        return build_market_pulse(stocks, as_of=as_of)

    def connect(self) -> None:
        """Connect the existing FYERS client to the controlled 59-stock universe."""
        self.tick_client.connect(self.symbols)

    def keep_running(self) -> None:
        self.tick_client.keep_running()

    def close(self) -> None:
        """Flush the current minute; the FYERS SDK owns socket shutdown."""
        self.candle_builder.flush()
