"""Historical analogue selection without future-information leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math


@dataclass(frozen=True)
class AnalogueEvent:
    event_id: str
    event_date: str
    feature_vector: tuple[float, ...]
    one_r: int
    two_r: int
    three_r: int
    stop: int


@dataclass(frozen=True)
class AnalogueSummary:
    count: int
    one_r_probability: float
    two_r_probability: float
    three_r_probability: float
    stop_probability: float
    weighted_score: float


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("analogue feature vectors must have equal length")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def summarize_analogues(current_features: Sequence[float], historical: Sequence[AnalogueEvent], k: int = 10) -> AnalogueSummary:
    if k <= 0:
        raise ValueError("k must be positive")
    if not historical:
        return AnalogueSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ranked = sorted(historical, key=lambda event: _distance(current_features, event.feature_vector))[:k]
    weights = [1.0 / max(_distance(current_features, event.feature_vector), 1e-9) for event in ranked]
    total = sum(weights)
    one = sum(w * event.one_r for w, event in zip(weights, ranked)) / total
    two = sum(w * event.two_r for w, event in zip(weights, ranked)) / total
    three = sum(w * event.three_r for w, event in zip(weights, ranked)) / total
    stop = sum(w * event.stop for w, event in zip(weights, ranked)) / total
    return AnalogueSummary(len(ranked), one, two, three, stop, one + two + three - stop)
