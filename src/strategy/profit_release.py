"""Shadow-only profit release state machine for V2."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProfitAction(str, Enum):
    HOLD_FULL = "HOLD_FULL"
    HOLD = "HOLD"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    PROTECT_PROFIT = "PROTECT_PROFIT"
    TRAIL = "TRAIL"
    FULL_EXIT = "FULL_EXIT"


@dataclass(frozen=True)
class ProfitReleaseEvidence:
    current_r: float
    thesis_score: float
    momentum_score: float = 0.0
    volume_score: float = 0.0
    market_score: float = 0.0
    sector_score: float = 0.0
    prediction_probability: float | None = None
    t1_hit: bool = False
    t2_hit: bool = False
    invalidated: bool = False


@dataclass(frozen=True)
class ProfitReleaseDecision:
    action: ProfitAction
    protected_r: float | None
    reason_code: str


class ProfitReleaseEngine:
    """Deterministic shadow policy. It never sends or sizes a real order."""

    @staticmethod
    def decide(e: ProfitReleaseEvidence) -> ProfitReleaseDecision:
        if e.invalidated or e.thesis_score < 30:
            return ProfitReleaseDecision(ProfitAction.FULL_EXIT, None, "THESIS_INVALIDATED")
        if e.t2_hit and e.thesis_score < 60:
            return ProfitReleaseDecision(ProfitAction.PARTIAL_EXIT, 1.0, "T2_WITH_WEAK_THESIS")
        if e.t1_hit and e.thesis_score < 50:
            return ProfitReleaseDecision(ProfitAction.PARTIAL_EXIT, 0.5, "T1_WITH_WEAK_THESIS")
        if e.current_r >= 2.0 and e.thesis_score >= 75 and e.momentum_score >= 60:
            return ProfitReleaseDecision(ProfitAction.TRAIL, 1.0, "STRONG_TREND_TRAILING")
        if e.current_r >= 1.0 and e.thesis_score >= 60:
            return ProfitReleaseDecision(ProfitAction.PROTECT_PROFIT, 0.5, "PROFIT_PROTECTION")
        if e.current_r > 0:
            return ProfitReleaseDecision(ProfitAction.HOLD, None, "POSITIVE_R_HOLD")
        return ProfitReleaseDecision(ProfitAction.HOLD_FULL, None, "NO_PROFIT_EVENT")
