import numpy as np
import pandas as pd

from src.prediction.entry_prediction_model_v1 import CALIBRATION_GAP, CALIBRATION_SESSIONS, _calibration_split, _folds, _top_lift


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


def test_calibration_split_is_session_aware_and_purged():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rows = []
    for d in dates:
        rows.extend([
            {"trade_date": d, "label_target_before_stop": 0, "ticker_symbol": "A"},
            {"trade_date": d, "label_target_before_stop": 1, "ticker_symbol": "B"},
        ])
    df = pd.DataFrame(rows)
    train_idx, calibration_idx = _calibration_split(df)
    train_dates = df.iloc[train_idx]["trade_date"]
    calibration_dates = df.iloc[calibration_idx]["trade_date"]
    assert CALIBRATION_GAP == 15
    assert len(calibration_dates.dt.normalize().unique()) == CALIBRATION_SESSIONS
    assert (calibration_dates.min() - train_dates.max()).days >= CALIBRATION_GAP + 1
