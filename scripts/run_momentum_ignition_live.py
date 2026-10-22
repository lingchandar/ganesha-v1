"""Run the read-only daily-prediction + live-confirmation engine."""
from __future__ import annotations

from statistics import mean

from src.backtest.data_loader import HistoricalDataLoader
from src.market_pulse.live_scanner import LiveMarketScanner, NSE_59_SYMBOLS
from src.market_pulse.momentum_ignition_live import MomentumIgnitionLiveEngine


def build_volume_baseline(loader: HistoricalDataLoader) -> dict[str, float]:
    """Build a 20-session daily-volume baseline for the controlled universe."""
    baseline: dict[str, float] = {}
    for symbol in NSE_59_SYMBOLS:
        frame = loader.stock_candles.get(symbol)
        if frame is None or "volume" not in frame.columns:
            continue
        values = [float(v) for v in frame["volume"].tail(20) if float(v) > 0]
        if values:
            baseline[symbol] = mean(values)
    return baseline


def main() -> None:
    loader = HistoricalDataLoader()
    loader.load_data()
    baseline = build_volume_baseline(loader)

    scanner = LiveMarketScanner(
        symbols=NSE_59_SYMBOLS,
        baseline_daily_volume=baseline,
    )
    engine = MomentumIgnitionLiveEngine(scanner, top_n=10)
    watchlist = engine.refresh_watchlist()

    print("\nNEXT-SESSION MOMENTUM IGNITION WATCHLIST")
    print(
        watchlist[
            [
                "trade_date",
                "ticker_symbol",
                "ignition_probability",
                "return_1d",
                "volume_ratio_20d",
                "volume_zscore_20d",
            ]
        ].to_string(index=False)
    )
    print("\nLive confirmation is read-only. ENTRY is printed only when a watchlist stock triggers the independent live momentum event detector.\n")

    try:
        engine.connect()
        engine.keep_running()
    except KeyboardInterrupt:
        print("\nStopping Ganesha live momentum ignition scanner...")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
