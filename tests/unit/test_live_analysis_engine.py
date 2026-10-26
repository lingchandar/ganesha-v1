from datetime import datetime, timezone, timedelta

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.setups.swing_setups import SwingSetupSignal


class FakeScanner:
    def __init__(self):
        self.calls = []

    def scan_all_setups(self, ticker_symbol, df):
        self.calls.append((ticker_symbol, df.copy()))
        return [
            SwingSetupSignal(
                ticker_symbol=ticker_symbol,
                setup_type="TREND_PULLBACK",
                entry_price=2100.0,
                stop_loss=2050.0,
                target_price=2200.0,
                risk_reward_ratio=2.0,
                risk_per_share=50.0,
                atr_14=20.0,
                volume_surge_multiple=1.2,
                delivery_ratio=0.0,
                invalidation_level=2075.0,
                rationale="test",
            ),
            SwingSetupSignal(
                ticker_symbol=ticker_symbol,
                setup_type="BREAKOUT_RETEST",
                entry_price=2100.0,
                stop_loss=2050.0,
                target_price=2200.0,
                risk_reward_ratio=2.0,
                risk_per_share=50.0,
                atr_14=20.0,
                volume_surge_multiple=1.2,
                delivery_ratio=0.0,
                invalidation_level=2075.0,
                rationale="test",
            ),
        ]


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


def build_engine(scanner=None):
    return LiveAnalysisEngine(
        min_candles=60,
        scanner=scanner or FakeScanner(),
        analysis_timeframe_minutes=15,
    )


def feed_until_first_scan(engine):
    results = []
    for i in range(15 * 60):
        results = engine.on_candle(make_candle(i))
    return results


def test_scans_only_after_60_complete_15_minute_bars():
    scanner = FakeScanner()
    engine = build_engine(scanner)

    results = feed_until_first_scan(engine)

    assert len(results) == 1
    assert results[0].setup_type == "TREND_PULLBACK"
    assert len(scanner.calls) == 1
    assert len(scanner.calls[0][1]) == 60
    assert scanner.calls[0][0] == "NSE:TCS-EQ"


def test_same_analysis_bar_does_not_scan_twice():
    scanner = FakeScanner()
    engine = build_engine(scanner)

    feed_until_first_scan(engine)
    calls_before = len(scanner.calls)

    assert engine.on_candle(make_candle(15 * 60 - 1)) == []
    assert len(scanner.calls) == calls_before


def test_live_analysis_never_registers_a_trade():
    engine = build_engine()

    results = feed_until_first_scan(engine)

    assert results
    assert not engine.trade_manager.has_active_trade("NSE:TCS-EQ")


def test_non_trend_pullback_setups_are_filtered():
    class OtherSetupScanner:
        def scan_all_setups(self, ticker_symbol, df):
            return [
                SwingSetupSignal(
                    ticker_symbol=ticker_symbol,
                    setup_type="BREAKOUT_RETEST",
                    entry_price=2100.0,
                    stop_loss=2050.0,
                    target_price=2200.0,
                    risk_reward_ratio=2.0,
                    risk_per_share=50.0,
                    atr_14=20.0,
                    volume_surge_multiple=1.2,
                    delivery_ratio=0.0,
                    invalidation_level=2075.0,
                    rationale="test",
                )
            ]

    engine = build_engine(OtherSetupScanner())

    assert feed_until_first_scan(engine) == []
    assert not engine.trade_manager.has_active_trade("NSE:TCS-EQ")
