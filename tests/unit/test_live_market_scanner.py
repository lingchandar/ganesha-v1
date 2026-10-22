from datetime import datetime, timedelta, timezone

from src.ingestion.fyers_realtime import LiveMarketTick
from src.market_pulse.live_scanner import LiveMarketScanner, NSE_59_SYMBOLS


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
        timestamp=datetime(2026, 10, 14, 4, 0, tzinfo=UTC) + timedelta(minutes=minute),
        raw_message={},
    )


def test_controlled_universe_contains_59_symbols():
    assert len(NSE_59_SYMBOLS) == 59
    assert len(set(NSE_59_SYMBOLS)) == 59
    assert all(symbol.startswith("NSE:") and symbol.endswith("-EQ") for symbol in NSE_59_SYMBOLS)


def test_scanner_builds_candles_and_detects_early_momentum_event():
    scanner = LiveMarketScanner(
        symbols=("NSE:KOTAKBANK-EQ",),
        baseline_daily_volume={"NSE:KOTAKBANK-EQ": 1_000_000},
    )

    # Feed one-minute ticks. The tick for the next minute closes the previous candle.
    prices = [100.00, 100.05, 100.10, 100.25, 100.45, 100.70, 101.00, 101.20]
    volumes = [1_000, 2_000, 3_000, 4_000, 5_000, 7_000, 15_000, 100_000]
    for minute, (price, volume) in enumerate(zip(prices, volumes)):
        scanner.process_tick(tick("NSE:KOTAKBANK-EQ", minute, price, volume))

    snapshot = scanner.snapshots()[0]

    assert scanner.buffer.count("NSE:KOTAKBANK-EQ") == 7
    assert snapshot.price == 101.0
    assert snapshot.change_pct > 0
    assert snapshot.return_3m_pct > 0.5
    assert snapshot.session_volume_ratio > 1.75
    assert snapshot.momentum_score > 0
    assert snapshot.money_flow_score > 0
    assert snapshot.momentum_event is True


def test_market_pulse_ranks_live_scanner_snapshot():
    scanner = LiveMarketScanner(
        symbols=("NSE:KOTAKBANK-EQ", "NSE:TCS-EQ"),
        baseline_daily_volume={
            "NSE:KOTAKBANK-EQ": 1_000_000,
            "NSE:TCS-EQ": 1_000_000,
        },
    )

    for minute in range(3):
        scanner.process_tick(tick("NSE:KOTAKBANK-EQ", minute, 100 + minute * 0.5, 10_000 + minute * 10_000))
        scanner.process_tick(tick("NSE:TCS-EQ", minute, 100 + minute * 0.05, 10_000 + minute * 1_000))

    pulse = scanner.market_pulse(as_of=datetime(2026, 10, 14, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))))

    assert len(pulse.stocks) == 2
    assert pulse.top_momentum[0].ticker_symbol == "NSE:KOTAKBANK-EQ"
