"""Run Ganesha's read-only live pipeline for TCS.

Usage:
    .venv/bin/python scripts/live_tcs_monitor.py

The process only reads FYERS market data. It never places orders.
"""

from src.ingestion.live_swing_monitor import LiveSwingMonitor


SYMBOLS = ["NSE:TCS-EQ"]


def on_signal(symbol: str, signals: list) -> None:
    if not signals:
        print(f"[SCAN] {symbol}: no setup")
        return

    print(f"[SIGNAL] {symbol}: {len(signals)} setup(s)")
    for signal in signals:
        print(signal)


if __name__ == "__main__":
    monitor = LiveSwingMonitor(SYMBOLS, signal_handler=on_signal)
    counts = monitor.warm_up(lookback_days=10)
    print("Warm-up candles:", counts)
    if counts.get(SYMBOLS[0], 0) < monitor.analysis.min_candles:
        raise SystemExit(
            "Insufficient historical warm-up for the live scanner."
        )

    print("Starting FYERS live stream...")
    monitor.start()
    monitor.candle_engine.run_forever()
