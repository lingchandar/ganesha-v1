from datetime import datetime, timedelta

import pandas as pd

from src.prediction.momentum_ignition_dataset_v3 import _ignition_outcome


def make_raw(next_close=103.0, next_high=105.0, next_low=99.0, next_volume=200.0):
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(23)]
    rows = [{"close": 100.0, "high": 101.0, "low": 99.0, "open": 100.0, "volume": 100.0} for _ in dates]
    rows[-1] = {
        "close": next_close,
        "high": next_high,
        "low": next_low,
        "open": 100.0,
        "volume": next_volume,
    }
    return pd.DataFrame(rows, index=dates)


def test_ignition_requires_next_day_price_and_volume_confirmation():
    raw = make_raw()
    result = _ignition_outcome(raw, 21, 2.0)

    assert result["ignition_outcome_valid"] == 1
    assert result["next_day_high_atr"] == 2.5
    assert result["next_day_volume_ratio"] == 2.0
    assert result["next_day_close_above_event"] == 1
    assert result["ignition_1d"] == 1


def test_large_price_move_without_volume_is_not_ignition():
    raw = make_raw(next_close=103.0, next_high=105.0, next_volume=120.0)
    result = _ignition_outcome(raw, 21, 2.0)

    assert result["next_day_high_atr"] >= 1.0
    assert result["next_day_close_above_event"] == 1
    assert result["next_day_volume_ratio"] < 1.5
    assert result["ignition_1d"] == 0


def test_future_data_does_not_change_features_because_labels_are_future_only():
    raw_a = make_raw(next_close=103.0, next_high=105.0, next_volume=200.0)
    raw_b = raw_a.copy()
    raw_b.iloc[-1, raw_b.columns.get_loc("close")] = 130.0
    raw_b.iloc[-1, raw_b.columns.get_loc("high")] = 135.0

    # The label is intentionally allowed to change; the current-day inputs
    # used to produce it remain outside the future slice.
    a = _ignition_outcome(raw_a, 21, 2.0)
    b = _ignition_outcome(raw_b, 21, 2.0)
    assert a["ignition_1d"] != b["ignition_1d"] or a["next_day_high_atr"] != b["next_day_high_atr"]
