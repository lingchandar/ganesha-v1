import numpy as np
import pandas as pd

from src.prediction.entry_prediction_model_v1 import CALIBRATION_GAP, _folds, _make_model, _top_lift


def test_top_lift_rewards_ranked_positive_examples():
    y = pd.Series([1, 1, 0, 0, 0, 0])
    probability = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert _top_lift(y, probability, 0.33) == 3.0


def test_folds_keep_purge_between_train_and_validation():
    dates = pd.date_range("2024-01-01", periods=420, freq="D")
    rows = []
    for d in dates:
        rows.extend([
            {"trade_date": d, "label_target_before_stop": 0, "ticker_symbol": "A"},
            {"trade_date": d, "label_target_before_stop": 1, "ticker_symbol": "B"},
        ])
    df = pd.DataFrame(rows)
    folds = _folds(df)
    train, valid = folds[0]
    assert train["trade_date"].max() < valid["trade_date"].min()
    assert (valid["trade_date"].min() - train["trade_date"].max()).days >= 15


def test_calibration_uses_same_purge_gap_as_outer_validation():
    model = _make_model()
    assert CALIBRATION_GAP == 15
    assert model.cv.n_splits == 3
    assert model.cv.gap == CALIBRATION_GAP
