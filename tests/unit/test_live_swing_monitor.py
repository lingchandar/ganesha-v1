from src.ingestion.live_swing_monitor import LiveSwingMonitor


def test_monitor_requires_symbols():
    try:
        LiveSwingMonitor([])
    except ValueError as exc:
        assert "FYERS symbol" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_monitor_deduplicates_symbols():
    monitor = LiveSwingMonitor(["NSE:TCS-EQ", "NSE:TCS-EQ"])
    assert monitor.symbols == ("NSE:TCS-EQ",)


def test_monitor_daily_exit_evaluation_uses_completed_daily_candle():
    from datetime import date

    from src.risk.exit_engine import ExitEvaluationResult
    from src.setups.swing_setups import SwingSetupSignal

    monitor = LiveSwingMonitor(["NSE:TCS-EQ"])
    signal = SwingSetupSignal(
        ticker_symbol="NSE:TCS-EQ",
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
    )
    monitor.analysis.trade_manager.open_trade(signal, date(2026, 9, 30))

    class FakeClient:
        def fetch_historical_candles(self, symbol, from_date, to_date, resolution):
            assert symbol == "NSE:TCS-EQ"
            assert from_date == to_date == date(2026, 10, 1)
            assert resolution == "D"
            import pandas as pd
            return pd.DataFrame([{
                "timestamp": pd.Timestamp("2026-10-01", tz="Asia/Kolkata"),
                "open_price": 2105.0,
                "high_price": 2190.0,
                "low_price": 2090.0,
                "close_price": 2180.0,
            }])

        def close(self):
            pass

    results = monitor.evaluate_daily_exits(date(2026, 10, 1), FakeClient())

    assert isinstance(results["NSE:TCS-EQ"], ExitEvaluationResult)
    assert not results["NSE:TCS-EQ"].is_exit_triggered
    assert results["NSE:TCS-EQ"].holding_days_elapsed == 1
