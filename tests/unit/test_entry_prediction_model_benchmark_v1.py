import pandas as pd

from scripts.benchmark_entry_prediction_models_v1 import FEATURES, _calibration_indices, _models


def test_algorithm_benchmark_has_multiple_global_candidates():
    names = list(_models())
    assert len(names) >= 4
    assert "hist_gradient_boosting" in names
    assert "extra_trees" in names
    assert "random_forest" in names
    assert "logistic_regression" in names


def test_benchmark_features_are_the_existing_global_v1_features():
    assert FEATURES
    assert "rsi_14" in FEATURES
    assert "setup_trend_pullback" in FEATURES


def test_benchmark_calibration_split_is_session_aware_and_purged():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    frame = pd.DataFrame({"trade_date": dates})
    train_idx, calibration_idx = _calibration_indices(frame)
    assert len(train_idx) > 0
    assert len(calibration_idx) == 60
    assert frame.iloc[train_idx]["trade_date"].max() < frame.iloc[calibration_idx]["trade_date"].min()
