import pandas as pd

from src.prediction.momentum_event_dataset_v2 import _event_outcome


def _raw():
    # 20 completed sessions are required for the pre-event volume baseline.
    # The event is evaluated on the final historical row (index=20).
    rows = {
        "open": [100.0] * 24,
        "high": [100.5] * 20 + [100.5, 102.0, 104.0, 105.0],
        "low": [99.5] * 24,
        "close": [100.0] * 24,
        "volume": [1000.0] * 20 + [1000.0, 1100.0, 1600.0, 900.0],
    }
    return pd.DataFrame(rows)


def test_event_requires_large_upside_and_volume_confirmation():
    raw = _raw()
    raw.loc[21:, "close"] = [101.0, 103.0, 104.0]
    result = _event_outcome(raw, index=20, atr_value=1.0)
    assert result["event_outcome_valid"] == 1
    assert result["abnormal_up_move_3d"] == 1
    assert result["event_volume_confirmed_3d"] == 1
    assert result["early_momentum_event_3d"] == 1
    assert result["event_days_to_peak_3d"] == 3


def test_event_is_not_positive_without_volume_confirmation():
    raw = _raw()
    raw.loc[21:, "close"] = [101.0, 103.0, 104.0]
    raw.loc[21:, "volume"] = 1000.0
    result = _event_outcome(raw, index=20, atr_value=1.0)
    assert result["abnormal_up_move_3d"] == 1
    assert result["event_volume_confirmed_3d"] == 0
    assert result["early_momentum_event_3d"] == 0


def test_event_does_not_use_event_day_or_future_close_as_features():
    raw = _raw()
    baseline = _event_outcome(raw, index=20, atr_value=1.0)
    raw.loc[20, "high"] = 1000.0
    raw.loc[20, "low"] = 1.0
    raw.loc[21:, "close"] = [101.0, 103.0, 104.0]
    changed_current = _event_outcome(raw, index=20, atr_value=1.0)
    assert baseline == changed_current
