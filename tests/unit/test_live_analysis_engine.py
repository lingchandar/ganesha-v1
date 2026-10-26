from datetime import date, datetime, time, timedelta, timezone

from src.ingestion.live_analysis_engine import LiveAnalysisEngine
from src.ingestion.one_minute_candle_builder import OneMinuteCandle
from src.setups.swing_setups import SwingSetupSignal


class FakeScanner:
    def __init__(self):
        self.calls = []

    def scan_all_setups(self, ticker_symbol, df):
        self.calls.append((ticker_symbol, df.copy()))
        return [
            SwingSetupSignal(ticker_symbol=ticker_symbol, setup_type="TREND_PULLBACK", entry_price=2100.0, stop_loss=2050.0, target_price=2200.0, risk_reward_ratio=2.0, risk_per_share=50.0, atr_14=20.0, volume_surge_multiple=1.2, delivery_ratio=0.0, invalidation_level=2075.0, rationale="test"),
            SwingSetupSignal(ticker_symbol=ticker_symbol, setup_type="BREAKOUT_RETEST", entry_price=2100.0, stop_loss=2050.0, target_price=2200.0, risk_reward_ratio=2.0, risk_per_share=50.0, atr_14=20.0, volume_surge_multiple=1.2, delivery_ratio=0.0, invalidation_level=2075.0, rationale="test"),
        ]


def make_candle(day: date, minute: int, price: float = 2000.0):
    timestamp = datetime.combine(day, time(9, 59), tzinfo=timezone.utc) + timedelta(minutes=minute)
    return OneMinuteCandle(symbol="NSE:TCS-EQ", timestamp=timestamp, open_price=price, high_price=price + 2, low_price=price - 1, close_price=price + 1, volume_traded=1000, tick_count=10)


def build_engine(scanner=None):
    engine = LiveAnalysisEngine(scanner=scanner or FakeScanner())
    for i in range(299):
        engine.analysis_buffers.add(OneMinuteCandle(symbol="NSE:TCS-EQ", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i), open_price=1900.0 + i, high_price=1902.0 + i, low_price=1899.0 + i, close_price=1901.0 + i, volume_traded=100000, tick_count=1))
    return engine


def test_daily_setup_is_not_scanned_before_session_close():
    scanner = FakeScanner()
    engine = build_engine(scanner)
    assert engine.on_candle(make_candle(date(2026, 10, 26), -1)) == []
    assert scanner.calls == []


def test_completed_daily_bar_scans_only_trend_pullback():
    scanner = FakeScanner()
    engine = build_engine(scanner)
    engine.on_candle(make_candle(date(2026, 10, 26), -1))
    result = engine.on_candle(make_candle(date(2026, 10, 26), 0))
    assert len(result) == 1
    assert result[0].setup_type == "TREND_PULLBACK"
    assert len(scanner.calls) == 1
    assert len(scanner.calls[0][1]) == 300
    assert not engine.trade_manager.has_active_trade("NSE:TCS-EQ")


def test_same_completed_session_cannot_scan_twice():
    scanner = FakeScanner()
    engine = build_engine(scanner)
    engine.on_candle(make_candle(date(2026, 10, 26), -1))
    engine.on_candle(make_candle(date(2026, 10, 26), 0))
    calls_before = len(scanner.calls)
    assert engine.on_candle(make_candle(date(2026, 10, 26), 1)) == []
    assert len(scanner.calls) == calls_before


def test_next_session_finalizes_previous_session_if_close_tick_was_missing():
    scanner = FakeScanner()
    engine = build_engine(scanner)
    engine.on_candle(make_candle(date(2026, 10, 26), -1))
    result = engine.on_candle(make_candle(date(2026, 10, 27), -1))
    assert len(result) == 1
    assert result[0].setup_type == "TREND_PULLBACK"


def test_non_trend_pullback_setups_are_filtered():
    class OtherSetupScanner:
        def scan_all_setups(self, ticker_symbol, df):
            return [SwingSetupSignal(ticker_symbol=ticker_symbol, setup_type="BREAKOUT_RETEST", entry_price=2100.0, stop_loss=2050.0, target_price=2200.0, risk_reward_ratio=2.0, risk_per_share=50.0, atr_14=20.0, volume_surge_multiple=1.2, delivery_ratio=0.0, invalidation_level=2075.0, rationale="test")]
    engine = build_engine(OtherSetupScanner())
    engine.on_candle(make_candle(date(2026, 10, 26), -1))
    assert engine.on_candle(make_candle(date(2026, 10, 26), 0)) == []
    assert not engine.trade_manager.has_active_trade("NSE:TCS-EQ")
