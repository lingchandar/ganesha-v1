import numpy as np

from scripts.evaluate_entry_prediction_calibration_oos_v1 import _ece, _metrics


def test_ece_is_zero_for_perfectly_calibrated_groups():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert _ece(y, p) == 0.0


def test_metrics_preserve_ranking_information():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = _metrics(y, p)
    assert result["auc"] == 1.0
    assert result["top_10pct_lift"] == 2.0
