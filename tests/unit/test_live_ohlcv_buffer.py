from datetime import datetime, timezone

from src.ingestion.live_ohlcv_buffer import LiveOHLCVBuffer
from src.ingestion.one_minute_candle_builder import OneMinuteCandle


def candle(minute, close):
    return OneMinuteCandle(
        symbol="NSE:TCS-EQ",
        timestamp=datetime(2026, 9, 29, 9, minute, tzinfo=timezone.utc),
        open_price=close - 1,
        high_price=close + 1,
        low_price=close - 2,
        close_price=close,
        volume_traded=100,
        tick_count=5,
    )


def test_buffer_keeps_bounded_ordered_history():
    buffer = LiveOHLCVBuffer(max_candles=2)
    buffer.add(candle(15, 2000))
    buffer.add(candle(16, 2001))
    buffer.add(candle(17, 2002))

    df = buffer.get("NSE:TCS-EQ")

    assert len(df) == 2
    assert df["timestamp"].tolist() == [
        datetime(2026, 9, 29, 9, 16, tzinfo=timezone.utc),
        datetime(2026, 9, 29, 9, 17, tzinfo=timezone.utc),
    ]
    assert df["close_price"].tolist() == [2001, 2002]


def test_buffer_ignores_duplicate_or_late_candle():
    buffer = LiveOHLCVBuffer()
    buffer.add(candle(15, 2000))
    buffer.add(candle(15, 1990))
    buffer.add(candle(14, 1980))

    assert buffer.count("NSE:TCS-EQ") == 1
    assert buffer.get("NSE:TCS-EQ").iloc[0]["close_price"] == 2000
