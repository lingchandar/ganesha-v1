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
        # 03:45 UTC = 09:15 IST, so the fixture starts exactly at market open.
        timestamp=datetime(2026, 10, 14, 3, 45, tzinfo=UTC) + timedelta(minutes=minute),
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

    # Two ticks per minute create realistic candle ranges. The final minute is
    # closed by the next minute's tick, so the scanner evaluates seven candles.
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
        scanner.process_tick(tick("NSE:KOTAKBANK-EQ", minute, price, volume))

    snapshot = scanner.snapshots()[0]

    assert scanner.buffer.count("NSE:KOTAKBANK-EQ") == 7
    assert snapshot.price == 101.0
    assert snapshot.change_pct > 0
    assert snapshot.return_3m_pct > 0.5
    assert snapshot.session_volume_ratio > 1.75
    assert snapshot.range_ratio > 1.25
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
