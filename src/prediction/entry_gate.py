"""Centralized, fail-closed entry authorization for Ganesha v1.

Momentum Ignition only produces candidates.  This gate is the final decision
boundary before an order may be created.  It deliberately refuses to approve
an entry when the Entry Prediction model has not passed its persisted
out-of-sample production gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EntryGateInput:
    momentum_rank: Optional[int]
    live_confirmation: bool
    model_available: bool
    model_production_ready: bool
    prediction_probability: Optional[float]
    trade_eligible: bool
    risk_allowed: bool
    position_size: float


@dataclass(frozen=True)
class EntryGateDecision:
    action: str
    approved: bool
    reason: str


class EntryGate:
    """Fail-closed final entry gate.

    Ranking is used for candidate selection, not as a substitute for model
    validation.  Ranks 1-5 may become entry candidates once every downstream
    condition is satisfied.  Ranks 6-10 remain WATCH even when confirmed.
    """

    MAX_ENTRY_RANK = 5

    def evaluate(self, inputs: EntryGateInput) -> EntryGateDecision:
        if inputs.momentum_rank is None:
            return EntryGateDecision("NO_ENTRY", False, "NOT_ON_IGNITION_WATCHLIST")
        if inputs.momentum_rank < 1:
            return EntryGateDecision("NO_ENTRY", False, "INVALID_WATCHLIST_RANK")
        if not inputs.live_confirmation:
            return EntryGateDecision("WATCH", False, "LIVE_CONFIRMATION_MISSING")
        if inputs.momentum_rank > self.MAX_ENTRY_RANK:
            return EntryGateDecision("WATCH", False, "RANK_OUTSIDE_ENTRY_CANDIDATE_WINDOW")
        if not inputs.model_available:
            return EntryGateDecision("NO_ENTRY", False, "ENTRY_MODEL_UNAVAILABLE")
        if not inputs.model_production_ready:
            return EntryGateDecision("NO_ENTRY", False, "ENTRY_MODEL_PRODUCTION_GATE_FAILED")
        if inputs.prediction_probability is None:
            return EntryGateDecision("NO_ENTRY", False, "ENTRY_PREDICTION_MISSING")
        if not 0.0 <= inputs.prediction_probability <= 1.0:
            return EntryGateDecision("NO_ENTRY", False, "INVALID_ENTRY_PROBABILITY")
        if not inputs.trade_eligible:
            return EntryGateDecision("NO_ENTRY", False, "TRADE_ELIGIBILITY_FAILED")
        if not inputs.risk_allowed:
            return EntryGateDecision("NO_ENTRY", False, "RISK_GATE_FAILED")
        if inputs.position_size <= 0:
            return EntryGateDecision("NO_ENTRY", False, "INVALID_POSITION_SIZE")
        return EntryGateDecision("ENTRY", True, "ALL_ENTRY_GATES_PASSED")
