import pandas as pd

from scripts.diagnose_entry_prediction_v1 import _daily_top_k, _metric


def test_metric_detects_perfect_ranking():
    frame = pd.DataFrame({
        "prediction": [0.9, 0.1, 0.8, 0.2],
        "label_target_before_stop": [1, 0, 1, 0],
    })
    result = _metric(frame)
    assert result["auc"] == 1.0


def test_daily_top_k_uses_daily_ranking():
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3),
        "prediction": [0.9, 0.2, 0.1, 0.8, 0.3, 0.2],
        "label_target_before_stop": [1, 0, 0, 0, 1, 0],
    })
    rate, lift, days = _daily_top_k(frame, 1)
    assert rate == 0.5
    assert lift == 1.5
    assert days == 2
