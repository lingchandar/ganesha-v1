import pandas as pd

from scripts.diagnose_entry_prediction_stable_edges_v1 import discover_candidates


def _frame(rows=240):
    data = []
    for i in range(rows):
        label = 1 if i % 4 == 0 else 0
        probability = 0.9 if label else 0.1
        data.append(
            {
                "trade_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "ticker_symbol": f"T{i:03d}",
                "setup_type": "TREND_PULLBACK",
                "trend_regime": "bullish",
                "momentum_regime": "positive",
                "volatility_regime": "high",
                "label_target_before_stop": label,
                "probability": probability,
                "fold": (i // 40) + 1,
            }
        )
    return pd.DataFrame(data)


def test_discover_candidates_requires_pooled_edge_and_stability():
    report = discover_candidates(_frame())
    assert not report.empty
    assert report.iloc[0]["pooled_auc"] > 0.99
    assert report.iloc[0]["pooled_top_10_lift"] > 1.0
    assert report.iloc[0]["stable_auc_folds"] >= 4
    assert report.iloc[0]["stable_lift_folds"] >= 4


def test_discover_candidates_rejects_low_positive_rate_groups():
    frame = _frame()
    frame["label_target_before_stop"] = 0
    frame.loc[:9, "label_target_before_stop"] = 1
    frame["probability"] = frame["label_target_before_stop"].astype(float)
    assert discover_candidates(frame).empty
