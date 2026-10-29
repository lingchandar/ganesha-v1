"""Deterministic 0-100 thesis-strength scorer for V2 shadow evaluation.

This scorer is evidence aggregation only. It never authorizes an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class ThesisEvidence:
    setup_quality: float = 0.0
    market_regime: float = 0.0
    sector_regime: float = 0.0
    price_volume_confirmation: float = 0.0
    prediction_probability: Optional[float] = None
    earnings_context: float = 0.0
    risk_quality: float = 0.0


@dataclass(frozen=True)
class ThesisStrengthResult:
    score: float
    reason_codes: tuple[str, ...]
    components: dict[str, float]


class ThesisStrengthScorer:
    """Weighted evidence score; weights are explicit for later ablation."""

    WEIGHTS = {
        "setup_quality": 0.20,
        "market_regime": 0.15,
        "sector_regime": 0.15,
        "price_volume_confirmation": 0.20,
        "prediction_probability": 0.15,
        "earnings_context": 0.05,
        "risk_quality": 0.10,
    }

    @classmethod
    def score(cls, evidence: ThesisEvidence) -> ThesisStrengthResult:
        raw = {
            "setup_quality": _clamp(evidence.setup_quality),
            "market_regime": _clamp(evidence.market_regime),
            "sector_regime": _clamp(evidence.sector_regime),
            "price_volume_confirmation": _clamp(evidence.price_volume_confirmation),
            "prediction_probability": _clamp((evidence.prediction_probability or 0.0) * 100.0),
            "earnings_context": _clamp(evidence.earnings_context),
            "risk_quality": _clamp(evidence.risk_quality),
        }
        score = sum(raw[key] * weight for key, weight in cls.WEIGHTS.items())
        reasons: list[str] = []
        if raw["setup_quality"] < 50: reasons.append("WEAK_SETUP")
        if raw["market_regime"] < 50: reasons.append("WEAK_MARKET_REGIME")
        if raw["sector_regime"] < 50: reasons.append("WEAK_SECTOR_REGIME")
        if raw["price_volume_confirmation"] < 50: reasons.append("WEAK_PRICE_VOLUME_CONFIRMATION")
        if evidence.prediction_probability is None: reasons.append("PREDICTION_UNAVAILABLE")
        elif evidence.prediction_probability < 0.20: reasons.append("LOW_MODEL_PROBABILITY")
        if raw["risk_quality"] < 50: reasons.append("WEAK_RISK_QUALITY")
        if score >= 75: reasons.append("THESIS_STRONG")
        elif score >= 60: reasons.append("THESIS_ACCEPTABLE")
        else: reasons.append("THESIS_WEAK")
        return ThesisStrengthResult(round(score, 2), tuple(reasons), raw)
