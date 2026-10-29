"""Fail-closed V2 shadow orchestration boundary.

This module produces an explainable research decision. It deliberately does
not call brokers or EntryGate and cannot place an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.strategy.state_machine import GaneshaStateMachine, StateEvidence, TradeState
from src.strategy.thesis_strength import ThesisEvidence, ThesisStrengthScorer


@dataclass(frozen=True)
class ShadowDecision:
    state: TradeState
    thesis_score: float
    allowed_action: str
    reason_codes: tuple[str, ...]
    model_probability: Optional[float]


class ShadowDecisionOrchestrator:
    def __init__(self) -> None:
        self.state_machine = GaneshaStateMachine()

    def evaluate(self, state_evidence: StateEvidence, thesis_evidence: ThesisEvidence, model_probability: Optional[float] = None) -> ShadowDecision:
        thesis = ThesisStrengthScorer.score(thesis_evidence)
        transition = self.state_machine.evaluate(
            StateEvidence(
                setup_valid=state_evidence.setup_valid,
                setup_ready=state_evidence.setup_ready,
                entry_confirmed=state_evidence.entry_confirmed,
                entry_executed=state_evidence.entry_executed,
                profit_management_active=state_evidence.profit_management_active,
                thesis_score=thesis.score,
                invalidated=state_evidence.invalidated,
                exit_pending=state_evidence.exit_pending,
                exit_profit=state_evidence.exit_profit,
                exit_loss=state_evidence.exit_loss,
                time_expired=state_evidence.time_expired,
                cooldown=state_evidence.cooldown,
            )
        )
        if transition.current in {TradeState.EXITED_PROFIT, TradeState.EXITED_LOSS, TradeState.TIME_EXPIRED, TradeState.INVALIDATED}:
            action = "SHADOW_EXIT"
        elif transition.current == TradeState.ENTRY_CONFIRMED:
            action = "SHADOW_ENTRY_CONFIRMED"
        elif transition.current in {TradeState.PROFIT_MANAGEMENT, TradeState.THESIS_WEAKENING}:
            action = "SHADOW_MANAGE"
        else:
            action = "SHADOW_WATCH"
        return ShadowDecision(transition.current, thesis.score, action, (transition.reason_code, *thesis.reason_codes), model_probability)
