import numpy as np
import pandas as pd

from src.prediction.momentum_event_model import _build_folds, _top_decile_metrics


def test_build_folds_purges_before_each_validation_window():
    dates = pd.date_range("2023-01-01", periods=440, freq="D")
    folds = _build_folds(dates, n_folds=4)
    assert len(folds) == 4
    for _, train_idx, val_idx in folds:
        assert train_idx[-1] < val_idx[0]
        assert val_idx[0] - train_idx[-1] >= 5
        assert len(val_idx) == 60


def test_top_decile_lift_is_relative_to_base_rate():
    y = pd.Series([0] * 8 + [1] * 2)
    probability = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99, 0.98])
    rate, lift = _top_decile_metrics(y, probability)
    assert rate == 1.0
    assert lift == 5.0
