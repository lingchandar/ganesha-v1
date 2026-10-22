import numpy as np
import pandas as pd

from src.prediction.momentum_ignition_model_v1 import (
    FEATURES,
    LABEL,
    PURGE_SESSIONS,
    _build_features,
    _rank_lift,
)


def test_model_features_exclude_future_labels():
    future_labels = {
        "ignition_1d",
        "continuation_3d",
        "next_day_up_move_atr",
        "next_day_high_atr",
        "next_day_range_atr",
        "next_day_volume_ratio",
        "next_day_close_above_event",
    }
    assert LABEL == "ignition_1d"
    assert future_labels.isdisjoint(FEATURES)


def test_rank_lift_returns_top_fraction():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.01, 0.02, 0.03, 0.04, 0.95, 0.90, 0.80, 0.20, 0.10, 0.05])
    lift, count = _rank_lift(y, probabilities, 0.20)
    assert count == 2
    assert lift > 3.0


def test_build_features_uses_only_declared_columns():
    frame = pd.DataFrame({column: [float(index), float(index + 1)] for index, column in enumerate(FEATURES)})
    frame["ignition_1d"] = [0, 1]
    result = _build_features(frame)
    assert list(result.columns) == FEATURES
    assert "ignition_1d" not in result.columns
    assert PURGE_SESSIONS == 1
