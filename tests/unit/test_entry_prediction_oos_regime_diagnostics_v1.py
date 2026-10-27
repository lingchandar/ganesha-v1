import pandas as pd

from scripts.diagnose_entry_prediction_oos_v1 import (
    _group_metrics,
    _momentum_regime,
    _trend_regime,
    _volatility_regime,
)


def test_trend_regime_classifies_bullish_bearish_and_mixed():
    frame = pd.DataFrame(
        {
            "price_vs_ema200_pct": [0.1, -0.1, 0.1],
            "ema20_vs_ema50_pct": [0.05, -0.05, -0.02],
            "ema50_vs_ema200_pct": [0.03, -0.03, 0.02],
        }
    )
    assert _trend_regime(frame).tolist() == ["bullish", "bearish", "mixed"]


def test_momentum_regime_classifies_positive_negative_and_neutral():
    frame = pd.DataFrame(
        {
            "return_5d": [0.05, -0.05, 0.01],
            "plus_di_minus_di": [4.0, -4.0, 0.5],
            "rsi_14": [60.0, 40.0, 50.0],
        }
    )
    assert _momentum_regime(frame).tolist() == ["positive", "negative", "neutral"]


def test_volatility_regime_has_three_buckets():
    frame = pd.DataFrame({"atr_pct": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    assert set(_volatility_regime(frame)) == {"low", "normal", "high"}


def test_group_metrics_returns_ranking_metrics_for_sufficient_binary_group():
    frame = pd.DataFrame(
        {
            "label_target_before_stop": [1, 1, 0, 0] * 30,
            "probability": [0.9, 0.8, 0.2, 0.1] * 30,
        }
    )
    metrics = _group_metrics(frame)
    assert metrics["rows"] == 120
    assert metrics["auc"] > 0.99
    assert metrics["top_10_lift"] > 1.0


def test_group_metrics_suppresses_unstable_small_groups():
    frame = pd.DataFrame(
        {
            "label_target_before_stop": [1, 0] * 20,
            "probability": [0.9, 0.1] * 20,
        }
    )
    metrics = _group_metrics(frame)
    assert metrics["rows"] == 40
    assert metrics["auc"] is None
    assert metrics["top_10_lift"] is None
