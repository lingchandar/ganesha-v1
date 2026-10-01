from datetime import datetime, timezone

from src.ingestion.fyers_realtime import LiveMarketTick
from src.ingestion.live_candle_engine import LiveCandleEngine


def test_tick_handler_reaches_candle_builder():
    candles = []
    engine = LiveCandleEngine(candle_handler=candles.append)

    engine._on_tick(
        LiveMarketTick(
            symbol="NSE:TCS-EQ",
            ltp=2000.0,
            open_price=2000.0,
            high_price=2000.0,
            low_price=2000.0,
            previous_close=1990.0,
            volume_traded=100,
            timestamp=datetime(2026, 9, 29, 9, 15, 10, tzinfo=timezone.utc),
            raw_message={},
        )
    )
    engine._on_tick(
        LiveMarketTick(
            symbol="NSE:TCS-EQ",
            ltp=2005.0,
            open_price=2000.0,
            high_price=2005.0,
            low_price=2000.0,
            previous_close=1990.0,
            volume_traded=150,
            timestamp=datetime(2026, 9, 29, 9, 16, 1, tzinfo=timezone.utc),
            raw_message={},
        )
    )

    assert len(candles) == 1
    assert candles[0].symbol == "NSE:TCS-EQ"
    assert candles[0].open_price == 2000.0
    assert candles[0].close_price == 2000.0
    # The first cumulative session-volume tick is not attributable to 09:15.
    assert candles[0].volume_traded == 0
