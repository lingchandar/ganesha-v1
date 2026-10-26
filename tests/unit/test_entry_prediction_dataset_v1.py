import pandas as pd

from src.prediction.entry_prediction_dataset_v1 import _label_trade


class Signal:
    entry_price = 100.0
    stop_loss = 98.0
    target_price = 104.0


def test_target_before_stop():
    features = pd.DataFrame([
        {"high_price": 102.0, "low_price": 99.0},
        {"high_price": 105.0, "low_price": 101.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=2)
    assert result["label_target_before_stop"] == 1
    assert result["mfe_r"] == 2.5


def test_stop_before_target():
    features = pd.DataFrame([
        {"high_price": 101.0, "low_price": 97.0},
        {"high_price": 105.0, "low_price": 99.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=2)
    assert result["label_target_before_stop"] == 0
    assert result["days_to_stop"] == 1


def test_same_bar_is_conservatively_adverse():
    features = pd.DataFrame([
        {"high_price": 105.0, "low_price": 97.0},
    ])
    result = _label_trade(features, 0, Signal(), horizon=1)
    assert result["label_target_before_stop"] == 0
