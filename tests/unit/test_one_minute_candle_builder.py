from datetime import datetime, timezone

from src.ingestion.fyers_realtime import LiveMarketTick
from src.ingestion.one_minute_candle_builder import OneMinuteCandleBuilder


def tick(ts, ltp, volume):
    return LiveMarketTick(
        symbol="NSE:TCS-EQ",
        ltp=ltp,
        open_price=2000.0,
        high_price=2010.0,
        low_price=1990.0,
        previous_close=2001.0,
        volume_traded=volume,
        timestamp=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
        raw_message={},
    )


def test_builds_ohlcv_and_uses_cumulative_volume_delta():
    completed = []
    builder = OneMinuteCandleBuilder(completed.append)

    builder.update(tick("2026-09-29T09:15:05", 2000.0, 100))
    builder.update(tick("2026-09-29T09:15:20", 2004.0, 150))
    builder.update(tick("2026-09-29T09:15:59", 1998.0, 230))

    candle = builder.update(tick("2026-09-29T09:16:01", 2002.0, 250))

    assert candle is not None
    assert candle.timestamp.isoformat() == "2026-09-29T09:15:00+00:00"
    assert candle.open_price == 2000.0
    assert candle.high_price == 2004.0
    assert candle.low_price == 1998.0
    assert candle.close_price == 1998.0
    # FYERS volume is cumulative; 230 - 100 = 130 traded during this minute.
    assert candle.volume_traded == 130
    assert candle.tick_count == 3
    assert completed == [candle]


def test_ignores_late_tick():
    builder = OneMinuteCandleBuilder()
    builder.update(tick("2026-09-29T09:15:30", 2000.0, 100))
    builder.update(tick("2026-09-29T09:16:00", 2005.0, 120))

    assert builder.update(tick("2026-09-29T09:15:45", 1990.0, 130)) is None


def test_flush_returns_open_candle():
    builder = OneMinuteCandleBuilder()
    builder.update(tick("2026-09-29T09:15:30", 2000.0, 100))
    candles = builder.flush()

    assert len(candles) == 1
    assert candles[0].close_price == 2000.0
    # The first cumulative session-volume tick cannot be attributed to this minute.
    assert candles[0].volume_traded == 0


def test_missing_ltp_or_timestamp_is_ignored():
    builder = OneMinuteCandleBuilder()

    missing_ltp = LiveMarketTick(
        symbol="NSE:TCS-EQ",
        ltp=None,
        open_price=None,
        high_price=None,
        low_price=None,
        previous_close=None,
        volume_traded=100,
        timestamp=datetime.now(timezone.utc),
        raw_message={},
    )
    assert builder.update(missing_ltp) is None
