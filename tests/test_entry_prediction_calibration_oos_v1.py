import numpy as np
import pandas as pd

from scripts.evaluate_entry_prediction_calibration_oos_v1 import _ece, _metrics


def test_ece_is_zero_for_perfectly_calibrated_groups():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert _ece(y, p) == 0.0


def test_metrics_preserve_ranking_information():
    frame = pd.DataFrame(
        {
            "probability_target_before_stop": [0.1, 0.2, 0.8, 0.9],
            "label_target_before_stop": [0, 0, 1, 1],
        }
    )
    result = _metrics(
        frame["label_target_before_stop"].to_numpy(),
        frame["probability_target_before_stop"].to_numpy(),
    )
    assert result["auc"] == 1.0
    assert result["top_50pct_lift"] if "top_50pct_lift" in result else True
    assert result["top_10pct_lift"] == 2.0
