"""Run Ganesha's read-only live daily swing validation for TCS.

Usage:
    .venv/bin/python scripts/live_tcs_monitor.py

The process only reads FYERS market data. It never places orders.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.live_swing_monitor import LiveSwingMonitor


SYMBOLS = ["NSE:TCS-EQ"]


def on_signal(symbol: str, signals: list) -> None:
    if not signals:
        print(f"[SCAN] {symbol}: no TREND_PULLBACK setup")
        return

    print(f"[SIGNAL] {symbol}: {len(signals)} validated setup(s)")
    for signal in signals:
        print(signal)


if __name__ == "__main__":
    monitor = LiveSwingMonitor(SYMBOLS, signal_handler=on_signal)
    counts = monitor.warm_up(lookback_days=450)
    print("Warm-up daily candles:", counts)
    if counts.get(SYMBOLS[0], 0) < monitor.analysis.min_candles:
        raise SystemExit("Insufficient historical warm-up for the validated daily swing scanner.")

    print("Starting FYERS live stream...")
    monitor.start()
    try:
        monitor.candle_engine.run_forever()
    except KeyboardInterrupt:
        print("\nLive monitor stopped by user.")
