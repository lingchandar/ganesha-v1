"""Leakage-safe forward labels for V2 historical prediction research."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import pandas as pd


@dataclass(frozen=True)
class MultiLevelLabel:
    setup_valid: int
    favorable_entry: int
    t1_hit: int
    t2_hit: int
    one_r_before_stop: int
    two_r_before_stop: int
    three_r_before_stop: int
    first_event: str
    forward_days: int


def _first_hit(frame: pd.DataFrame, entry: float, stop: float, target: float) -> str:
    for row in frame.itertuples(index=False):
        low = float(row.low); high = float(row.high)
        if low <= stop and high >= target:
            return "STOP_ASSUMED_FIRST"
        if low <= stop:
            return "STOP"
        if high >= target:
            return "TARGET"
    return "TIMEOUT"


def label_forward_window(candles: pd.DataFrame, entry: float, stop: float, targets: Sequence[float], max_days: int = 15) -> MultiLevelLabel:
    """Label only candles strictly after the signal row supplied by caller."""
    if entry <= stop:
        raise ValueError("entry must be above stop for long setup")
    if not targets:
        raise ValueError("at least one target is required")
    frame = candles.iloc[:max_days].copy()
    required = {"high", "low"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing candle columns: {sorted(missing)}")

    risk = entry - stop
    levels = [entry + risk * float(r) for r in (1.0, 2.0, 3.0)]
    supplied_t1 = float(targets[0])
    supplied_t2 = float(targets[1] if len(targets) > 1 else targets[0])
    first = _first_hit(frame, entry, stop, supplied_t2)
    return MultiLevelLabel(
        setup_valid=1,
        favorable_entry=int(float(frame.iloc[0]["high"]) >= entry if not frame.empty else False),
        t1_hit=int(bool((frame["high"] >= supplied_t1).any())),
        t2_hit=int(bool((frame["high"] >= supplied_t2).any())),
        one_r_before_stop=int(_first_hit(frame, entry, stop, levels[0]) == "TARGET"),
        two_r_before_stop=int(_first_hit(frame, entry, stop, levels[1]) == "TARGET"),
        three_r_before_stop=int(_first_hit(frame, entry, stop, levels[2]) == "TARGET"),
        first_event=first,
        forward_days=int(len(frame)),
    )
