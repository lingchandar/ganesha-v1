"""Bridge the daily Momentum Ignition model to live FYERS confirmation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.market_pulse.live_scanner import LiveMarketScanner, LiveScannerSnapshot
from src.prediction.fresh_momentum_ignition import build_fresh_next_session_watchlist


@dataclass(frozen=True)
class MomentumIgnitionLiveDecision:
    """Real-time decision for a stock already ranked by the daily model."""

    ticker_symbol: str
    ignition_probability: float
    watchlist_rank: Optional[int]
    price: float
    return_3m_pct: float
    return_5m_pct: float
    session_volume_ratio: float
    range_ratio: float
    momentum_event: bool
    action: str


class MomentumIgnitionLiveEngine:
    """Combine fresh daily predictions with live intraday confirmation.

    The daily model creates the watchlist from fresh FYERS completed daily
    candles. The live scanner independently confirms whether the predicted
    ignition is actually starting. No order execution is performed.
    """

    STATUS_LOG_INTERVAL_SECONDS = 60.0

    def __init__(
        self,
        scanner: LiveMarketScanner,
        *,
        watchlist: Optional[pd.DataFrame] = None,
        top_n: int = 10,
        status_logging: bool = True,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.scanner = scanner
        self.top_n = top_n
        self.status_logging = status_logging
        self._watchlist: dict[str, tuple[float, int]] = {}
        self._latest_decisions: dict[str, MomentumIgnitionLiveDecision] = {}
        self._last_status_log = 0.0
        self.scanner.set_snapshot_handler(self._on_live_snapshot)
        if watchlist is not None:
            self.set_watchlist(watchlist)

    def _on_live_snapshot(self, snapshot: LiveScannerSnapshot) -> None:
        decision = self.evaluate_snapshot(snapshot)
        self._latest_decisions[snapshot.ticker_symbol] = decision
        if decision.action == "ENTRY":
            print(
                "MOMENTUM_IGNITION_ENTRY "
                f"ticker={decision.ticker_symbol} "
                f"probability={decision.ignition_probability:.4f} "
                f"price={decision.price:.2f} "
                f"return_3m={decision.return_3m_pct:.2f}% "
                f"return_5m={decision.return_5m_pct:.2f}% "
                f"volume_ratio={decision.session_volume_ratio:.2f}x "
                f"range_ratio={decision.range_ratio:.2f}x"
            )
        self._maybe_log_status()

    def _maybe_log_status(self) -> None:
        """Print a compact watchlist heartbeat at most once per minute."""
        if not self.status_logging or not self._watchlist:
            return
        now = time.monotonic()
        if now - self._last_status_log < self.STATUS_LOG_INTERVAL_SECONDS:
            return
        self._last_status_log = now

        decisions = [
            self._latest_decisions[symbol]
            for symbol in self._watchlist
            if symbol in self._latest_decisions
        ]
        if not decisions:
            return
        decisions.sort(key=lambda item: item.watchlist_rank or 999)

        print("\nLIVE MOMENTUM WATCHLIST")
        print("rank ticker         action  price       3m      5m    volume  range   ignition")
        print("---- ------------- ------- --------- ------- ------- ------- ------- --------")
        for decision in decisions:
            print(
                f"{decision.watchlist_rank:>4} "
                f"{decision.ticker_symbol.replace('NSE:', '').replace('-EQ', ''):<13} "
                f"{decision.action:<7} "
                f"{decision.price:>9.2f} "
                f"{decision.return_3m_pct:>6.2f}% "
                f"{decision.return_5m_pct:>6.2f}% "
                f"{decision.session_volume_ratio:>6.2f}x "
                f"{decision.range_ratio:>6.2f}x "
                f"{decision.ignition_probability:>7.2%}"
            )
        print()

    def refresh_watchlist(self) -> pd.DataFrame:
        """Build the next-session watchlist from fresh FYERS daily data."""
        watchlist = build_fresh_next_session_watchlist(top_n=self.top_n)
        self.set_watchlist(watchlist)
        return watchlist

    def set_watchlist(self, watchlist: pd.DataFrame) -> None