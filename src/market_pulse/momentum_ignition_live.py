"""Bridge the daily Momentum Ignition model to live FYERS confirmation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import pandas as pd

from src.market_pulse.live_scanner import LiveMarketScanner, LiveScannerSnapshot
from src.prediction.momentum_ignition_scorer import build_next_session_watchlist


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
    """Combine next-session daily predictions with live intraday confirmation.

    The daily model creates the watchlist. The live scanner confirms whether the
    predicted ignition is actually starting. No order execution is performed.
    """

    def __init__(
        self,
        scanner: LiveMarketScanner,
        *,
        watchlist: Optional[pd.DataFrame] = None,
        top_n: int = 10,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.scanner = scanner
        self.top_n = top_n
        self._watchlist: dict[str, tuple[float, int]] = {}
        if watchlist is not None:
            self.set_watchlist(watchlist)

    def refresh_watchlist(self) -> pd.DataFrame:
        """Build the latest completed-day next-session watchlist."""
        watchlist = build_next_session_watchlist(top_n=self.top_n)
        self.set_watchlist(watchlist)
        return watchlist

    def set_watchlist(self, watchlist: pd.DataFrame) -> None:
        required = {"ticker_symbol", "ignition_probability"}
        missing = required - set(watchlist.columns)
        if missing:
            raise ValueError(f"Watchlist missing required columns: {sorted(missing)}")

        ranked = watchlist.sort_values("ignition_probability", ascending=False).reset_index(drop=True)
        selected: dict[str, tuple[float, int]] = {}
        for rank, row in ranked.head(self.top_n).iterrows():
            symbol = str(row["ticker_symbol"])
            if symbol not in self.scanner.symbols:
                continue
            selected[symbol] = (float(row["ignition_probability"]), int(rank) + 1)
        self._watchlist = selected

    @property
    def watchlist_symbols(self) -> tuple[str, ...]:
        return tuple(self._watchlist)

    def process_tick(self, tick) -> Optional[MomentumIgnitionLiveDecision]:
        """Feed one FYERS tick and return the latest decision for its symbol."""
        self.scanner.process_tick(tick)
        snapshot = next(
            (item for item in self.scanner.snapshots() if item.ticker_symbol == tick.symbol),
            None,
        )
        if snapshot is None:
            return None
        return self.evaluate_snapshot(snapshot)

    def evaluate_snapshot(self, snapshot: LiveScannerSnapshot) -> MomentumIgnitionLiveDecision:
        model_signal = self._watchlist.get(snapshot.ticker_symbol)
        if model_signal is None:
            action = "AVOID"
            probability = 0.0
            rank = None
        else:
            probability, rank = model_signal
            # ENTRY means the model predicted the next-session ignition and the
            # independent live scanner confirms the move is starting now.
            action = "ENTRY" if snapshot.momentum_event else "WATCH"

        return MomentumIgnitionLiveDecision(
            ticker_symbol=snapshot.ticker_symbol,
            ignition_probability=probability,
            watchlist_rank=rank,
            price=snapshot.price,
            return_3m_pct=snapshot.return_3m_pct,
            return_5m_pct=snapshot.return_5m_pct,
            session_volume_ratio=snapshot.session_volume_ratio,
            range_ratio=snapshot.range_ratio,
            momentum_event=snapshot.momentum_event,
            action=action,
        )

    def decisions(self) -> tuple[MomentumIgnitionLiveDecision, ...]:
        """Return decisions for all stocks currently visible to the live scanner."""
        return tuple(
            self.evaluate_snapshot(snapshot)
            for snapshot in self.scanner.snapshots()
        )

    def entry_candidates(self) -> tuple[MomentumIgnitionLiveDecision, ...]:
        """Return only model-backed stocks with live momentum confirmation."""
        return tuple(decision for decision in self.decisions() if decision.action == "ENTRY")

    def connect(self) -> None:
        self.scanner.connect()

    def keep_running(self) -> None:
        self.scanner.keep_running()

    def close(self) -> None:
        self.scanner.close()
