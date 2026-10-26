import pandas as pd

from src.prediction.entry_prediction_dataset_v1 import _label_trade


class Signal:
    entry_price = 100.0
    stop_loss = 98.0
    target_price = 104.0


def test_target_before_stop():
    features = pd.DataFrame([
        {"high_price": 100.0, "low_price": 100.0, "close_price": 100.0},
        {"high_price": 102.0, "low_price": 99.0, "close_price": 101.0},
        {"high_price": 105.0, "low_price": 101.0, "close_price": 104.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=2)
    assert result["label_target_before_stop"] == 1
    assert result["mfe_r"] == 2.5
    assert result["horizon_end_close"] is None
    assert result["horizon_end_r"] is None


def test_stop_before_target():
    features = pd.DataFrame([
        {"high_price": 100.0, "low_price": 100.0, "close_price": 100.0},
        {"high_price": 101.0, "low_price": 97.0, "close_price": 99.0},
        {"high_price": 105.0, "low_price": 99.0, "close_price": 104.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=2)
    assert result["label_target_before_stop"] == 0
    assert result["days_to_stop"] == 1
    assert result["horizon_end_close"] is None
    assert result["horizon_end_r"] is None


def test_same_bar_is_conservatively_adverse():
    features = pd.DataFrame([
        {"high_price": 100.0, "low_price": 100.0, "close_price": 100.0},
        {"high_price": 105.0, "low_price": 97.0, "close_price": 100.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=1)
    assert result["label_target_before_stop"] == 0
    assert result["days_to_stop"] == 1
    assert result["days_to_target"] == 1
    assert result["horizon_end_close"] is None
    assert result["horizon_end_r"] is None


def test_unresolved_records_horizon_end_close_and_r():
    features = pd.DataFrame([
        {"high_price": 100.0, "low_price": 100.0, "close_price": 100.0},
        {"high_price": 101.0, "low_price": 99.0, "close_price": 100.5},
        {"high_price": 102.0, "low_price": 99.5, "close_price": 101.0},
        {"high_price": 103.0, "low_price": 99.8, "close_price": 102.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=3)
    assert result["outcome_valid"] == 0
    assert result["horizon_end_close"] == 102.0
    assert result["horizon_end_r"] == 1.0


def test_unresolved_does_not_use_partial_horizon():
    features = pd.DataFrame([
        {"high_price": 100.0, "low_price": 100.0, "close_price": 100.0},
        {"high_price": 101.0, "low_price": 99.0, "close_price": 100.5},
        {"high_price": 102.0, "low_price": 99.5, "close_price": 101.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=3)
    assert result["outcome_valid"] == 0
    assert result["horizon_end_close"] is None
    assert result["horizon_end_r"] is None
