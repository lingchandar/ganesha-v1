"""Compare strategy components on untouched OOS metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AblationResult:
    name: str
    trade_count: int
    coverage: float
    expected_r: float
    profit_factor: float
    max_drawdown_r: float
    win_rate: float
    mean_mfe_r: float
    mean_mae_r: float


def rank_ablation(results: Iterable[AblationResult]) -> list[AblationResult]:
    """Rank by expected R, then profit factor, then lower drawdown.

    This is a reporting helper only; it does not promote a strategy.
    """
    return sorted(results, key=lambda item: (item.expected_r, item.profit_factor, item.max_drawdown_r), reverse=True)


def materially_improves(candidate: AblationResult, baseline: AblationResult, min_expected_r_delta: float = 0.05, min_profit_factor_delta: float = 0.10) -> bool:
    return (
        candidate.expected_r >= baseline.expected_r + min_expected_r_delta
        and candidate.profit_factor >= baseline.profit_factor + min_profit_factor_delta
        and candidate.max_drawdown_r >= baseline.max_drawdown_r
    )
