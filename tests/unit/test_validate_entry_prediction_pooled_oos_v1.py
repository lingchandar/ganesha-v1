import numpy as np
import pandas as pd

from scripts.validate_entry_prediction_pooled_oos_v1 import _pooled_metrics


def test_pooled_metrics_returns_expected_top_lifts():
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.99, 0.98, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])

    metrics = _pooled_metrics(y, p)

    assert metrics["rows"] == 10.0
    assert metrics["positive_rate"] == 0.2
    assert metrics["top_10pct_lift"] == 5.0
    assert metrics["top_5pct_lift"] == 5.0
    assert 0.0 <= metrics["brier"] <= 1.0
