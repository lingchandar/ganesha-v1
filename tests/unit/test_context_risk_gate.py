"""Unit tests for the point-in-time context risk gate."""
import numpy as np
import pandas as pd

from src.risk.context_risk_gate import ContextRiskGate


def _frame(closes, opens=None):
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = closes.copy()
    opens = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * 1.005,
            "low": np.minimum(opens, closes) * 0.995,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000),
        }
    )


def test_normal_context_is_allowed():
    closes = [100 + i * 0.2 for i in range(50)]
    decision = ContextRiskGate.evaluate(_frame(closes))
    assert decision.allowed is True
    assert decision.risk_level == "NORMAL"
    assert decision.reason == "NORMAL_CONTEXT"


def test_recent_large_return_blocks_trade():
    closes = [100 + i * 0.2 for i in range(49)] + [115]
    decision = ContextRiskGate.evaluate(_frame(closes))
    assert decision.allowed is False
    assert decision.risk_level == "HIGH"
    assert "RECENT_RETURN_SHOCK" in decision.reason


def test_recent_large_gap_blocks_trade():
    closes = [100.0] * 49 + [102.0]
    opens = [100.0] * 49 + [107.0]
    decision = ContextRiskGate.evaluate(_frame(closes, opens))
    assert decision.allowed is False
    assert "RECENT_GAP_SHOCK" in decision.reason


def test_insufficient_history_is_blocked():
    decision = ContextRiskGate.evaluate(_frame([100 + i for i in range(20)]))
    assert decision.allowed is False
    assert decision.risk_level == "UNKNOWN"
