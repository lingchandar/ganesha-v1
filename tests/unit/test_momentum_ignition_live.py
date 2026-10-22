from datetime import datetime, timedelta, timezone

import pandas as pd

from src.ingestion.fyers_realtime import LiveMarketTick
from src.market_pulse.live_scanner import LiveMarketScanner
from src.market_pulse.momentum_ignition_live import MomentumIgnitionLiveEngine

UTC = timezone.utc


def tick(symbol: str, minute: int, price: float, volume: int, previous_close: float = 100.0):
    return LiveMarketTick(
        symbol=symbol,
        ltp=price,
        open_price=100.0,
        high_price=price,
        low_price=price,
        previous_close=previous_close,
        volume_traded=volume,
        timestamp=datetime(2026, 10, 14, 3, 45, tzinfo=UTC) + timedelta(minutes=minute),
        raw_message={},
    )


def make_scanner():
    return LiveMarketScanner(
        symbols=("NSE:TRENT-EQ", "NSE:TCS-EQ"),
        baseline_daily_volume={
            "NSE:TRENT-EQ": 1_000_000,
            "NSE:TCS-EQ": 1_000_000,
        },
    )


def feed_confirmation(scanner):
    feed = [
        (0, 100.00, 1_000), (0, 100.20, 2_000),
        (1, 100.05, 3_000), (1, 100.25, 4_000),
        (2, 100.10, 5_000), (2, 100.30, 6_000),
        (3, 100.15, 7_000), (3, 100.35, 8_000),
        (4, 100.20, 9_000), (4, 100.40, 10_000),
        (5, 100.25, 12_000), (5, 100.45, 15_000),
        (6, 100.30, 18_000), (6, 101.00, 25_000),
        (7, 101.20, 120_000),
    ]
    for minute, price, volume in feed:
        scanner.process_tick(tick("NSE:TRENT-EQ", minute, price, volume))


def test_watchlist_signal_becomes_entry_after_live_confirmation():
    scanner = make_scanner()
    watchlist = pd.DataFrame(
        {
            "ticker_symbol": ["NSE:TRENT-EQ", "NSE:TCS-EQ"],
            "ignition_probability": [0.2947, 0.04],
        }
    )
    engine = MomentumIgnitionLiveEngine(scanner, watchlist=watchlist, top_n=2)

    feed_confirmation(scanner)
    decision = engine.evaluate_snapshot(scanner.snapshots()[0])

    assert decision.ticker_symbol == "NSE:TRENT-EQ"
    assert decision.ignition_probability == 0.2947
    assert decision.watchlist_rank == 1
    assert decision.momentum_event is True
    assert decision.action == "ENTRY"


def test_watchlist_without_confirmation_is_watch():
    scanner = make_scanner()
    watchlist = pd.DataFrame(
        {"ticker_symbol": ["NSE:TRENT-EQ"], "ignition_probability": [0.2947]}
    )
    engine = MomentumIgnitionLiveEngine(scanner, watchlist=watchlist)

    # One completed candle: model signal exists, but the live confirmation
    # conditions have not started yet.
    scanner.process_tick(tick("NSE:TRENT-EQ", 0, 100.10, 2_000))
    scanner.process_tick(tick("NSE:TRENT-EQ", 1, 100.12, 3_000))

    decision = engine.evaluate_snapshot(scanner.snapshots()[0])
    assert decision.action == "WATCH"
    assert decision.ignition_probability == 0.2947
    assert decision.momentum_event is False


def test_stock_outside_watchlist_is_avoid():
    scanner = make_scanner()
    watchlist = pd.DataFrame(
        {"ticker_symbol": ["NSE:TRENT-EQ"], "ignition_probability": [0.2947]}
    )
    engine = MomentumIgnitionLiveEngine(scanner, watchlist=watchlist)

    scanner.process_tick(tick("NSE:TCS-EQ", 0, 100.05, 2_000))
    scanner.process_tick(tick("NSE:TCS-EQ", 1, 100.06, 3_000))

    decision = engine.evaluate_snapshot(scanner.snapshots()[0])
    assert decision.ticker_symbol == "NSE:TCS-EQ"
    assert decision.action == "AVOID"
    assert decision.watchlist_rank is None
    assert decision.ignition_probability == 0.0
