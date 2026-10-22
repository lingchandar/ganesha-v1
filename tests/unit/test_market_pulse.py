from datetime import datetime
from zoneinfo import ZoneInfo

from src.market_pulse.engine import (
    MarketSession,
    MarketStockSnapshot,
    MoneyFlowSnapshot,
    MutualFundSnapshot,
    build_market_pulse,
    market_session,
)

IST = ZoneInfo("Asia/Kolkata")


def dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 10, 14, hour, minute, tzinfo=IST)


def stock(ticker: str, change: float, volume_ratio: float, momentum: float, flow: float):
    return MarketStockSnapshot(
        ticker_symbol=ticker,
        price=100.0,
        change_pct=change,
        volume=1_000_000,
        volume_ratio_20d=volume_ratio,
        momentum_score=momentum,
        money_flow_score=flow,
    )


def test_market_session_regular_nse_windows():
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


def test_rankings_are_computed_from_snapshot_data():
    stocks = (
        stock("NSE:A-EQ", 2.0, 1.2, 30.0, 4.0),
        stock("NSE:B-EQ", 1.0, 4.5, 20.0, 2.0),
        stock("NSE:C-EQ", -1.0, 2.0, 10.0, -3.0),
    )
    pulse = build_market_pulse(stocks, as_of=dt(10, 0))

    assert [x.ticker_symbol for x in pulse.top_momentum] == [
        "NSE:A-EQ",
        "NSE:B-EQ",
        "NSE:C-EQ",
    ]
    assert pulse.unusual_volume[0].ticker_symbol == "NSE:B-EQ"
    assert pulse.positive_money_flow[0].ticker_symbol == "NSE:A-EQ"


def test_flow_values_are_not_fabricated_when_missing():
    pulse = build_market_pulse(
        [stock("NSE:A-EQ", 1.0, 1.0, 1.0, 1.0)],
        as_of=dt(11, 0),
    )

    assert pulse.money_flow.fii_net is None
    assert pulse.money_flow.dii_net is None
    assert pulse.mutual_funds.equity_net_flow is None
    assert pulse.mutual_funds.debt_net_flow is None
    assert pulse.mutual_funds.gold_net_flow is None


def test_external_flow_snapshots_are_preserved_with_source_dates():
    money = MoneyFlowSnapshot(fii_net=-123.4, dii_net=456.7, source_date="2026-10-13")
    funds = MutualFundSnapshot(
        equity_net_flow=100.0,
        debt_net_flow=-50.0,
        gold_net_flow=25.0,
        source_date="2026-09-30",
        is_intraday=False,
    )
    pulse = build_market_pulse(
        [stock("NSE:A-EQ", 1.0, 1.0, 1.0, 1.0)],
        as_of=dt(11, 0),
        money_flow=money,
        mutual_funds=funds,
    )

    assert pulse.money_flow == money
    assert pulse.mutual_funds == funds
