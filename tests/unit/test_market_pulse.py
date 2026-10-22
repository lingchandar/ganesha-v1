from datetime import datetime
from zoneinfo import ZoneInfo

from src.market_pulse.engine import (
    MarketSession,
    MarketStockSnapshot,
    build_market_pulse,
    market_session,
)

IST = ZoneInfo("Asia/Kolkata")


def dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 10, 14, hour, minute, tzinfo=IST)


def test_market_session_regular_hours():
    assert market_session(dt(8, 59)) == MarketSession.CLOSED
    assert market_session(dt(9, 0)) == MarketSession.PRE_OPEN
    assert market_session(dt(9, 14)) == MarketSession.PRE_OPEN
    assert market_session(dt(9, 15)) == MarketSession.INTRADAY
    assert market_session(dt(15, 29)) == MarketSession.INTRADAY
    assert market_session(dt(15, 30)) == MarketSession.POST_CLOSE
    assert market_session(dt(16, 0)) == MarketSession.CLOSED


def test_weekend_is_closed():
    saturday = datetime(2026, 10, 17, 10, 0, tzinfo=IST)
    assert market_session(saturday) == MarketSession.CLOSED


def test_market_pulse_ranking():
    stocks = [
        MarketStockSnapshot("NSE:A-EQ", 100, volume_ratio_20d=4.0, momentum_score=90),
        MarketStockSnapshot("NSE:B-EQ", 100, volume_ratio_20d=2.0, momentum_score=70),
        MarketStockSnapshot("NSE:C-EQ", 100, volume_ratio_20d=3.0, momentum_score=80),
    ]
    pulse = build_market_pulse(stocks, as_of=dt(10, 0))

    assert pulse.session == MarketSession.INTRADAY
    assert pulse.top_momentum[0].ticker_symbol == "NSE:A-EQ"
    assert pulse.unusual_volume[0].ticker_symbol == "NSE:A-EQ"
    assert len(pulse.positive_money_flow) == 3
