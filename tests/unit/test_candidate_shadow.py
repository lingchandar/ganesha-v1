from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.prediction.candidate_shadow import CANDIDATE_VERSION, CandidateShadowPredictor
from src.prediction.engine import PredictionFeatureBuilder


class Signal:
    ticker_symbol = "NSE:TCS-EQ"
    setup_type = "TREND_PULLBACK"
    entry_price = 2100.0
    stop_loss = 2050.0
    target_price = 2200.0
    volume_surge_multiple = 1.2


def _features() -> pd.DataFrame:
    rows = []
    for i in range(25):
        rows.append({
            "close_price": 2100.0 + i,
            "atr_14": 20.0,
            "atr_sma_20": 19.0,
            "ema_20": 2095.0,
            "ema_50": 2080.0,
            "ema_200": 2000.0,
            "rsi_14": 60.0,
            "plus_di": 28.0,
            "minus_di": 20.0,
            "macd_hist": 3.0,
            "macd_hist_diff": 0.2,
            "adx_14": 25.0,
            "ema_50_slope_20": 0.01,
            "volume_traded": 100000,
        })
    return pd.DataFrame(rows)


def test_shadow_predictor_is_observation_only(tmp_path: Path):
    model = ExtraTreesClassifier(n_estimators=20, random_state=42)
    feature_frame = PredictionFeatureBuilder.build(Signal(), _features())
    X = pd.concat([feature_frame, feature_frame], ignore_index=True)
    y = [0, 1]
    model.fit(X, y)
    bundle = {
        "model_version": CANDIDATE_VERSION,
        "algorithm": "extra_trees",
        "feature_columns": PredictionFeatureBuilder.FEATURE_COLUMNS,
        "model": model,
        "production_ready": False,
        "promotion_status": "candidate_only",
    }
    path = tmp_path / "candidate.joblib"
    joblib.dump(bundle, path)

    predictor = CandidateShadowPredictor(path)
    result = predictor.predict(Signal(), _features())

    assert result.available is True
    assert result.model_version == CANDIDATE_VERSION
    assert 0.0 <= result.probability_target_before_stop <= 1.0
    assert result.reason == "SHADOW_PREDICTION"
