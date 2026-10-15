import pandas as pd

from src.prediction.momentum_event_dataset import _triple_barrier_outcome


def _raw():
    return pd.DataFrame(
        {
            "open": [100, 100, 101, 102, 103, 104, 105],
            "high": [101, 101, 103, 104, 105, 106, 107],
            "low": [99, 99, 100, 101, 102, 103, 104],
            "close": [100, 100, 102, 103, 104, 105, 106],
            "volume": [1000] * 7,
            "atr_14": [1.0] * 7,
        }
    )


def test_triple_barrier_marks_up_move_before_down_move():
    raw = _raw()
    result = _triple_barrier_outcome(raw, index=0, horizon=5, up_atr=1.5, down_atr=1.0)
    assert result["outcome_valid"] == 1
    assert result["hit_up_before_down_5d"] == 1
    assert result["days_to_up_5d"] == 2
    assert result["mfe_r_5d"] > 0


def test_triple_barrier_marks_adverse_move_when_down_barrier_hits_first():
    raw = _raw()
    raw.loc[1, "low"] = 98.5
    result = _triple_barrier_outcome(raw, index=0, horizon=5, up_atr=1.5, down_atr=1.0)
    assert result["hit_up_before_down_5d"] == 0


def test_outcome_only_uses_future_sessions():
    raw = _raw()
    baseline = _triple_barrier_outcome(raw, index=0, horizon=5)
    raw.loc[0, "high"] = 1000
    raw.loc[0, "low"] = 1
    changed_feature_candle = _triple_barrier_outcome(raw, index=0, horizon=5)
    assert baseline == changed_feature_candle
