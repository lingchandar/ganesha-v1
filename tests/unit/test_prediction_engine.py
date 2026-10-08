import pandas as pd
import pytest

from src.prediction.engine import EntryPredictionEngine, PredictionFeatureBuilder


def _features():
    rows = []
    for i in range(30):
        rows.append({
            "close_price": 100 + i,
            "atr_14": 2.0,
            "ema_20": 98 + i,
            "ema_50": 95 + i,
            "rsi_14": 60.0,
            "ema_50_slope_20": 0.2,
        })
    return pd.DataFrame(rows)


def _signal():
    return {
        "setup_type": "TREND_PULLBACK",
        "entry_price": 129.0,
        "stop_loss": 125.0,
        "target_price": 137.0,
        "risk_reward_ratio": 2.0,
        "volume_surge_multiple": 1.5,
    }


def test_prediction_is_unavailable_without_trained_model(tmp_path):
    engine = EntryPredictionEngine(tmp_path / "missing.joblib")
    result = engine.predict(_signal(), _features())
    assert not result.available
    assert result.reason == "MODEL_NOT_TRAINED"
    assert not engine.approve_entry(result)


def test_feature_builder_is_point_in_time_shaped():
    frame = PredictionFeatureBuilder.build(_signal(), _features())
    assert list(frame.columns) == PredictionFeatureBuilder.FEATURE_COLUMNS
    assert len(frame) == 1
    assert frame.isna().sum().sum() == 0
