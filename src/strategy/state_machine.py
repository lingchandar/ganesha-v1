from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TradeState(str, Enum):
    DISCOVERED = "DISCOVERED"
    QUALIFYING = "QUALIFYING"
    WATCH = "WATCH"
    READY = "READY"
    ENTRY_CONFIRMED = "ENTRY_CONFIRMED"
    ACTIVE = "ACTIVE"
    PROFIT_MANAGEMENT = "PROFIT_MANAGEMENT"
    THESIS_WEAKENING = "THESIS_WEAKENING"
    EXIT_PENDING = "EXIT_PENDING"
    EXITED_PROFIT = "EXITED_PROFIT"
    EXITED_LOSS = "EXITED_LOSS"
    TIME_EXPIRED = "TIME_EXPIRED"
    INVALIDATED = "INVALIDATED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class StateTransition:
    previous: TradeState
    current: TradeState
    reason_code: str


@dataclass(frozen=True)
class StateEvidence:
    setup_valid: bool = False
    setup_ready: bool = False
    entry_confirmed: bool = False
    entry_executed: bool = False
    profit_management_active: bool = False
    thesis_score: float = 0.0
    invalidated: bool = False
    exit_pending: bool = False
    exit_profit: bool = False
    exit_loss: bool = False
    time_expired: bool = False
    cooldown: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.thesis_score) <= 100.0:
            raise ValueError("thesis_score must be between 0 and 100")


class GaneshaStateMachine:
    """Fail-closed, deterministic V2 state machine for shadow evaluation."""

    def __init__(self, initial_state: TradeState = TradeState.DISCOVERED) -> None:
        self._state = initial_state
        self._history: list[StateTransition] = []

    @property
    def state(self) -> TradeState:
        return self._state

    @property
    def history(self) -> tuple[StateTransition, ...]:
        return tuple(self._history)

    def evaluate(self, evidence: StateEvidence) -> StateTransition:
        previous = self._state
        current, reason = self._next_state(previous, evidence)
        transition = StateTransition(previous, current, reason)
        if current != previous:
            self._state = current
            self._history.append(transition)
        return transition

    @staticmethod
    def _next_state(state: TradeState, e: StateEvidence) -> tuple[TradeState, str]:
        if state in {TradeState.EXITED_PROFIT, TradeState.EXITED_LOSS, TradeState.TIME_EXPIRED, TradeState.INVALIDATED}:
            return (TradeState.COOLDOWN, "COOLDOWN_AFTER_EXIT") if e.cooldown else (state, "TERMINAL_STATE")
        if state == TradeState.COOLDOWN:
            return (TradeState.DISCOVERED, "COOLDOWN_COMPLETE") if not e.cooldown else (state, "COOLDOWN_ACTIVE")
        if e.exit_profit:
            return TradeState.EXITED_PROFIT, "EXIT_PROFIT"
        if e.exit_loss:
            return TradeState.EXITED_LOSS, "EXIT_LOSS"
        if e.time_expired:
            return TradeState.TIME_EXPIRED, "TIME_EXPIRED"
        if e.invalidated:
            return TradeState.INVALIDATED, "SETUP_INVALIDATED"
        if e.exit_pending:
            return TradeState.EXIT_PENDING, "EXIT_CONDITION_PENDING"
        if state == TradeState.DISCOVERED:
            return (TradeState.QUALIFYING, "SETUP_DETECTED") if e.setup_valid else (state, "NO_VALID_SETUP")
        if state == TradeState.QUALIFYING:
            if not e.setup_valid:
                return TradeState.DISCOVERED, "SETUP_QUALIFICATION_FAILED"
            return (TradeState.READY, "SETUP_READY") if e.setup_ready else (TradeState.WATCH, "QUALIFYING_TO_WATCH")
        if state == TradeState.WATCH:
            if not e.setup_valid:
                return TradeState.DISCOVERED, "WATCH_INVALIDATED"
            return (TradeState.READY, "ENTRY_ZONE_READY") if e.setup_ready else (state, "WATCHING")
        if state == TradeState.READY:
            if not e.setup_valid:
                return TradeState.DISCOVERED, "READY_INVALIDATED"
            return (TradeState.ENTRY_CONFIRMED, "ENTRY_CONFIRMATION") if e.entry_confirmed else (state, "READY_WAITING")
        if state == TradeState.ENTRY_CONFIRMED:
            return (TradeState.ACTIVE, "ENTRY_EXECUTED") if e.entry_executed else (state, "ENTRY_PENDING")
        if state == TradeState.ACTIVE:
            if e.profit_management_active:
                return TradeState.PROFIT_MANAGEMENT, "PROFIT_MANAGEMENT_TRIGGERED"
            if e.thesis_score < 50.0:
                return TradeState.THESIS_WEAKENING, "THESIS_WEAKENING"
            return state, "POSITION_ACTIVE"
        if state == TradeState.PROFIT_MANAGEMENT:
            return (TradeState.THESIS_WEAKENING, "THESIS_WEAKENING") if e.thesis_score < 50.0 else (state, "PROFIT_MANAGEMENT_ACTIVE")
        if state == TradeState.THESIS_WEAKENING:
            return (TradeState.PROFIT_MANAGEMENT, "THESIS_RECOVERED") if e.thesis_score >= 50.0 and e.profit_management_active else (state, "THESIS_WEAKENING")
        if state == TradeState.EXIT_PENDING:
            return (TradeState.PROFIT_MANAGEMENT, "EXIT_CONDITION_CLEARED") if not e.exit_pending else (state, "EXIT_PENDING")
        return state, "NO_TRANSITION_RULE"

    def reset(self, state: TradeState = TradeState.DISCOVERED) -> None:
        self._state = state
        self._history.clear()

    def replay(self, evidence: Iterable[StateEvidence]) -> tuple[StateTransition, ...]:
        for snapshot in evidence:
            self.evaluate(snapshot)
        return self.history
