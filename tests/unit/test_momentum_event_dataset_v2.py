import pandas as pd

from src.prediction.momentum_event_dataset_v2 import _event_outcome


def _raw():
    return pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [100.5, 102.0, 104.0, 105.0, 106.0],
            "low": [99.5, 99.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1600.0, 900.0, 900.0],
        }
    )


def test_event_requires_large_upside_and_volume_confirmation():
    result = _event_outcome(_raw(), index=0, atr_value=1.0)
    assert result["event_outcome_valid"] == 1
    assert result["abnormal_up_move_3d"] == 1
    assert result["event_volume_confirmed_3d"] == 1
    assert result["early_momentum_event_3d"] == 1
    assert result["event_days_to_peak_3d"] == 3


def test_event_is_not_positive_without_volume_confirmation():
    raw = _raw()
    raw.loc[1:, "volume"] = 1000.0
    result = _event_outcome(raw, index=0, atr_value=1.0)
    assert result["abnormal_up_move_3d"] == 1
    assert result["event_volume_confirmed_3d"] == 0
    assert result["early_momentum_event_3d"] == 0


def test_event_does_not_use_event_day_or_future_close_as_features():
    raw = _raw()
    baseline = _event_outcome(raw, index=0, atr_value=1.0)
    raw.loc[0, "high"] = 1000.0
    raw.loc[0, "low"] = 1.0
    changed_current = _event_outcome(raw, index=0, atr_value=1.0)
    assert baseline == changed_current
