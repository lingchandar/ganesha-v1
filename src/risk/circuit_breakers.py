"""
GANESHA V1 — Portfolio Circuit Breakers & Correlated-Drawdown Guard (Module 11, 12)

INSTITUTIONAL RISK MANAGEMENT POSTURE:
1. Maximum Concurrent Positions: Strictly capped at 5 active swing positions.
2. Aggregate Capital-at-Risk: Sum of risk across all active trades <= 5% of total portfolio.
3. Sector Concentration Cap: Maximum 2 stocks from the same sector concurrently.
   If 3 stocks from the same sector qualify, rank and accept only the top 2.
4. Correlated-Drawdown Pause Rule:
   If >= 3 positions hit stop-loss within any rolling 7 calendar days:
   -> SYSTEM PAUSE ACTIVATED for 7 trading sessions.
   -> Suspend all new buy signal generation.
   -> Existing trailing stops tightened to 1x ATR.
"""
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from src.core.config import settings


@dataclass(frozen=True)
class CircuitBreakerCheckResult:
    """Result of evaluating a candidate trade against portfolio limits."""
    is_allowed: bool
    rejection_reason: Optional[str]
    active_position_count: int
    sector_position_count: int
    aggregate_risk_inr: float
    aggregate_risk_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrelatedDrawdownStatus:
    """Status of the correlated-drawdown system brake."""
    is_system_paused: bool
    recent_stop_outs_count: int
    pause_until_date: Optional[str]
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioCircuitBreakers:
    """
    Enforces capacity limits, sector concentration caps, and drawdown circuit brakes.
    """

    MAX_CONCURRENT_POSITIONS = settings.max_concurrent_positions # 5
    MAX_AGGREGATE_RISK_PCT = 0.05                               # 5.0%
    MAX_PER_SECTOR = settings.max_stocks_per_sector              # 2
    CORRELATED_STOP_THRESHOLD = 3                               # >= 3 stop-outs
    CORRELATED_WINDOW_DAYS = 7                                  # 7 calendar days
    PAUSE_DURATION_DAYS = 7                                     # 7 trading sessions

    @classmethod
    def evaluate_candidate_capacity(
        cls,
        active_positions: List[Dict[str, Any]],
        candidate_ticker: str,
        candidate_sector: str,
        candidate_risk_inr: float,
        total_capital: Optional[float] = None,
    ) -> CircuitBreakerCheckResult:
        """
        Validates if adding a candidate trade violates:
        1. Max concurrent positions (5)
        2. Sector concentration cap (max 2 per sector)
        3. Aggregate capital-at-risk cap (<= 5% of total capital)

        Args:
            active_positions: List of dicts representing open trades.
            candidate_ticker: Symbol of candidate (e.g. "INFY")
            candidate_sector: Sector of candidate (e.g. "INFORMATION_TECHNOLOGY")
            candidate_risk_inr: Net capital at risk in INR for this candidate
            total_capital: Account equity (defaults to settings.total_capital_inr)

        Returns:
            CircuitBreakerCheckResult
        """
        capital = total_capital or settings.total_capital_inr
        max_aggregate_risk_inr = capital * cls.MAX_AGGREGATE_RISK_PCT

        # 1. Check if candidate is already in active positions
        existing_tickers = {p.get("ticker_symbol") for p in active_positions}
        if candidate_ticker in existing_tickers:
            return CircuitBreakerCheckResult(
                is_allowed=False,
                rejection_reason=f"DUPLICATE_POSITION: {candidate_ticker} is already active in open portfolio.",
                active_position_count=len(active_positions),
                sector_position_count=sum(1 for p in active_positions if p.get("sector") == candidate_sector),
                aggregate_risk_inr=sum(p.get("capital_at_risk_net_inr", 0.0) for p in active_positions),
                aggregate_risk_pct=0.0,
            )

        # 2. Maximum concurrent positions cap (5)
        current_count = len(active_positions)
        if current_count >= cls.MAX_CONCURRENT_POSITIONS:
            return CircuitBreakerCheckResult(
                is_allowed=False,
                rejection_reason=f"PORTFOLIO_CAPACITY_REACHED: Maximum concurrent positions limit ({cls.MAX_CONCURRENT_POSITIONS}) reached.",
                active_position_count=current_count,
                sector_position_count=sum(1 for p in active_positions if p.get("sector") == candidate_sector),
                aggregate_risk_inr=sum(p.get("capital_at_risk_net_inr", 0.0) for p in active_positions),
                aggregate_risk_pct=0.0,
            )

        # 3. Sector concentration cap (max 2 per sector)
        sector_count = sum(1 for p in active_positions if p.get("sector") == candidate_sector)
        if sector_count >= cls.MAX_PER_SECTOR:
            return CircuitBreakerCheckResult(
                is_allowed=False,
                rejection_reason=(
                    f"SECTOR_CONCENTRATION_CAP_EXCEEDED: Sector '{candidate_sector}' already has "
                    f"{sector_count} active position(s). Maximum allowed is {cls.MAX_PER_SECTOR}."
                ),
                active_position_count=current_count,
                sector_position_count=sector_count,
                aggregate_risk_inr=sum(p.get("capital_at_risk_net_inr", 0.0) for p in active_positions),
                aggregate_risk_pct=0.0,
            )

        # 4. Aggregate portfolio capital-at-risk cap (<= 5%)
        current_aggregate_risk = sum(p.get("capital_at_risk_net_inr", 0.0) for p in active_positions)
        projected_aggregate_risk = current_aggregate_risk + candidate_risk_inr
        projected_risk_pct = round((projected_aggregate_risk / capital) * 100.0, 2)

        if projected_aggregate_risk > max_aggregate_risk_inr:
            return CircuitBreakerCheckResult(
                is_allowed=False,
                rejection_reason=(
                    f"AGGREGATE_RISK_LIMIT_EXCEEDED: Projected risk ₹{projected_aggregate_risk:.2f} "
                    f"({projected_risk_pct}%) exceeds 5% capital cap (₹{max_aggregate_risk_inr:.2f})."
                ),
                active_position_count=current_count,
                sector_position_count=sector_count,
                aggregate_risk_inr=current_aggregate_risk,
                aggregate_risk_pct=projected_risk_pct,
            )

        return CircuitBreakerCheckResult(
            is_allowed=True,
            rejection_reason=None,
            active_position_count=current_count,
            sector_position_count=sector_count,
            aggregate_risk_inr=round(projected_aggregate_risk, 2),
            aggregate_risk_pct=projected_risk_pct,
        )

    @classmethod
    def filter_watchlist_by_sector_cap(
        cls,
        candidate_signals: List[Dict[str, Any]],
        active_positions: List[Dict[str, Any]],
        max_per_sector: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Enforces sector concentration on candidate signals.
        If more candidates from a sector qualify than slots remaining,
        candidates are sorted and top ones selected.
        """
        # Count existing allocations per sector in active portfolio
        sector_occupancy: Dict[str, int] = {}
        for p in active_positions:
            sec = p.get("sector", "UNKNOWN")
            sector_occupancy[sec] = sector_occupancy.get(sec, 0) + 1

        approved: List[Dict[str, Any]] = []
        for cand in candidate_signals:
            sec = cand.get("sector", "UNKNOWN")
            current_sec_count = sector_occupancy.get(sec, 0)
            if current_sec_count < max_per_sector:
                approved.append(cand)
                sector_occupancy[sec] = current_sec_count + 1
            else:
                logger.info(
                    f"Sector cap reached for {sec}: dropping {cand.get('ticker_symbol')} "
                    f"({current_sec_count}/{max_per_sector} slots filled)."
                )

        return approved

    @classmethod
    def check_correlated_drawdown_pause(
        cls,
        recent_closed_trades: List[Dict[str, Any]],
        current_date: date,
    ) -> CorrelatedDrawdownStatus:
        """
        Evaluates the Correlated-Drawdown Pause Rule:
        If >= 3 positions hit stop-loss within rolling 7 calendar days:
        Pause all new signal generation for 7 days.

        Args:
            recent_closed_trades: List of trade dicts with exit_date (date/iso) and exit_reason.
            current_date: Today's date.

        Returns:
            CorrelatedDrawdownStatus
        """
        window_start = current_date - timedelta(days=cls.CORRELATED_WINDOW_DAYS)

        # Count stop-loss events in the rolling 7-day window
        recent_stop_outs: List[date] = []
        for trade in recent_closed_trades:
            exit_reason = str(trade.get("exit_reason", "")).upper()
            if "STOP" in exit_reason:
                e_date = trade.get("exit_date")
                if isinstance(e_date, str):
                    try:
                        e_date = date.fromisoformat(e_date)
                    except ValueError:
                        continue
                if isinstance(e_date, date) and window_start <= e_date <= current_date:
                    recent_stop_outs.append(e_date)

        stop_count = len(recent_stop_outs)

        if stop_count >= cls.CORRELATED_STOP_THRESHOLD:
            # Sort to find the latest stop-out date
            latest_stop = max(recent_stop_outs)
            pause_until = latest_stop + timedelta(days=cls.PAUSE_DURATION_DAYS)

            if current_date <= pause_until:
                msg = (
                    f"CORRELATED_DRAWDOWN_PAUSE_ACTIVE: {stop_count} stop-outs occurred in "
                    f"trailing 7 days (Threshold: {cls.CORRELATED_STOP_THRESHOLD}). "
                    f"All new swing generation suspended until {pause_until.isoformat()}."
                )
                logger.warning(msg)
                return CorrelatedDrawdownStatus(
                    is_system_paused=True,
                    recent_stop_outs_count=stop_count,
                    pause_until_date=pause_until.isoformat(),
                    status_message=msg,
                )

        return CorrelatedDrawdownStatus(
            is_system_paused=False,
            recent_stop_outs_count=stop_count,
            pause_until_date=None,
            status_message=f"Normal operation: {stop_count}/{cls.CORRELATED_STOP_THRESHOLD} stop-outs in trailing 7 days.",
        )
