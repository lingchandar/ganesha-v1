from datetime import datetime, timezone, timedelta

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.one_minute_candle_builder import OneMinuteCandle


class FakeScanner:
    def __init__(self):
        self.calls = []

    def scan_all_setups(self, ticker_symbol, df):
        self.calls.append((ticker_symbol, df.copy()))
        return ["signal"]


def make_candle(i):
    start = datetime(2026, 9, 29, 9, 15, tzinfo=timezone.utc)
    ts = start + timedelta(minutes=i)
    return OneMinuteCandle(
        symbol="NSE:TCS-EQ",
        timestamp=ts,
        open_price=2000 + i,
        high_price=2002 + i,
        low_price=1999 + i,
        close_price=2001 + i,
        volume_traded=1000,
        tick_count=10,
    )


def test_scans_only_after_60_complete_15_minute_bars():
    scanner = FakeScanner()
    engine = LiveAnalysisEngine(
        min_candles=60,
        scanner=scanner,
        analysis_timeframe_minutes=15,
    )

    results = []
    for i in range(15 * 60):
        results = engine.on_candle(make_candle(i))

    assert results == ["signal"]
    assert len(scanner.calls) == 1
    assert len(scanner.calls[0][1]) == 60
    assert scanner.calls[0][0] == "NSE:TCS-EQ"


def test_same_analysis_bar_does_not_scan_twice():
    scanner = FakeScanner()
    engine = LiveAnalysisEngine(
        min_candles=60,
        scanner=scanner,
        analysis_timeframe_minutes=15,
    )

    for i in range(15 * 60):
        engine.on_candle(make_candle(i))

    calls_before = len(scanner.calls)
    assert engine.on_candle(make_candle(15 * 60 - 1)) == []
    assert len(scanner.calls) == calls_before
