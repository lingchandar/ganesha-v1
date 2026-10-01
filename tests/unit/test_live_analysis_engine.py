from datetime import datetime, timezone

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.one_minute_candle_builder import OneMinuteCandle


class FakeScanner:
    def __init__(self):
        self.calls = []

    def scan_all_setups(self, ticker_symbol, df):
        self.calls.append((ticker_symbol, df.copy()))
        return ["signal"]


def make_candle(i):
    return OneMinuteCandle(
        symbol="NSE:TCS-EQ",
        timestamp=datetime(2026, 9, 29, 9, 15 + i, tzinfo=timezone.utc),
        open_price=2000 + i,
        high_price=2002 + i,
        low_price=1999 + i,
        close_price=2001 + i,
        volume_traded=1000,
        tick_count=10,
    )


def test_scans_only_after_warmup():
    scanner = FakeScanner()
    engine = LiveAnalysisEngine(min_candles=60, scanner=scanner)

    for i in range(59):
        assert engine.on_candle(make_candle(i)) == []

    signals = engine.on_candle(make_candle(59))

    assert signals == ["signal"]
    assert len(scanner.calls) == 1
    assert len(scanner.calls[0][1]) == 60
    assert scanner.calls[0][0] == "NSE:TCS-EQ"
