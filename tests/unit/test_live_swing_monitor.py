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
