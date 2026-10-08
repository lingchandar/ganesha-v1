"""Point-in-time trade prediction engine for Ganesha v1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


MODEL_VERSION = "entry_v1"
DEFAULT_MODEL_PATH = Path("models/entry_prediction_v1.joblib")


@dataclass(frozen=True)
class PredictionResult:
    available: bool
    probability_target_before_stop: Optional[float]
    expected_r: Optional[float]
    confidence: Optional[float]
    model_version: Optional[str]
    reason: str


def _signal_value(signal: Any, name: str, default: Any = None) -> Any:
    """Read a signal field from either a dict or SwingSetupSignal-like object."""
    if isinstance(signal, dict):
        return signal.get(name, default)
    return getattr(signal, name, default)


class PredictionFeatureBuilder:
    """Build features using only data known at the signal timestamp."""

    FEATURE_COLUMNS = [
        "rsi_14", "atr_pct", "volume_surge_multiple", "risk_reward_ratio",
        "distance_to_stop_atr", "distance_to_target_atr", "price_vs_ema20_pct",
        "price_vs_ema50_pct", "ema20_vs_ema50_pct", "ema50_slope_20",
        "return_5d", "return_20d", "volatility_20d", "setup_class",
    ]

    @classmethod
    def build(cls, signal: Any, df_features: pd.DataFrame) -> pd.DataFrame:
        if df_features.empty:
            raise ValueError("Feature frame cannot be empty")
        latest = df_features.iloc[-1]
        close = float(latest["close_price"])
        atr = max(float(latest["atr_14"]), 1e-9)
        ema20 = float(latest["ema_20"])
        ema50 = float(latest["ema_50"])
        close_series = pd.to_numeric(df_features["close_price"], errors="coerce")
        ret5 = float(close_series.pct_change(5).iloc[-1]) if len(close_series) > 5 else 0.0
        ret20 = float(close_series.pct_change(20).iloc[-1]) if len(close_series) > 20 else 0.0
        vol20 = float(close_series.pct_change().rolling(20).std().iloc[-1]) if len(close_series) > 20 else 0.0
        row = {
            "rsi_14": float(latest.get("rsi_14", 50.0)),
            "atr_pct": atr / close,
            "volume_surge_multiple": float(_signal_value(signal, "volume_surge_multiple", 1.0)),
            "risk_reward_ratio": float(_signal_value(signal, "risk_reward_ratio", 0.0)),
            "distance_to_stop_atr": (close - float(_signal_value(signal, "stop_loss"))) / atr,
            "distance_to_target_atr": (float(_signal_value(signal, "target_price")) - close) / atr,
            "price_vs_ema20_pct": (close / ema20) - 1.0 if ema20 else 0.0,
            "price_vs_ema50_pct": (close / ema50) - 1.0 if ema50 else 0.0,
            "ema20_vs_ema50_pct": (ema20 / ema50) - 1.0 if ema50 else 0.0,
            "ema50_slope_20": float(latest.get("ema_50_slope_20", 0.0)),
            "return_5d": ret5,
            "return_20d": ret20,
            "volatility_20d": vol20,
            "setup_class": float(_setup_id(str(_signal_value(signal, "setup_type", "")))),
        }
        return pd.DataFrame([row], columns=cls.FEATURE_COLUMNS)


def _setup_id(name: str) -> int:
    mapping = {
        "CLASSIC_RESISTANCE_BREAKOUT": 0,
        "BREAKOUT_RETEST": 1,
        "TREND_PULLBACK": 2,
        "DEMAND_SUPPORT_BOUNCE": 3,
        "TREND_CONTINUATION_FLAG": 4,
        "RS_LEADER_BREAKOUT": 5,
    }
    return mapping.get(name, -1)


class EntryPredictionEngine:
    """Loads a trained, temporally validated model for entry decisions."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._bundle: Optional[Dict] = None
        if self.model_path.exists():
            self._bundle = joblib.load(self.model_path)
            self._validate_bundle()

    @property
    def available(self) -> bool:
        return self._bundle is not None

    def _validate_bundle(self) -> None:
        if not isinstance(self._bundle, dict):
            raise ValueError("Prediction model bundle must be a dictionary")
        if self._bundle.get("model_version") != MODEL_VERSION:
            raise ValueError("Unsupported prediction model version")
        if not self._bundle.get("feature_columns"):
            raise ValueError("Prediction model bundle has no feature columns")
        if "model" not in self._bundle:
            raise ValueError("Prediction model bundle has no model")

    def predict(self, signal: Any, df_features: pd.DataFrame) -> PredictionResult:
        if not self.available:
            return PredictionResult(False, None, None, None, None, "MODEL_NOT_TRAINED")
        features = PredictionFeatureBuilder.build(signal, df_features)
        expected = self._bundle["feature_columns"]
        features = features.reindex(columns=expected, fill_value=0.0)
        probability = float(self._bundle["model"].predict_proba(features)[0, 1])
        expected_r = (probability * 2.0) + ((1.0 - probability) * -1.0)
        confidence = abs(probability - 0.5) * 2.0
        return PredictionResult(True, probability, expected_r, confidence, MODEL_VERSION, "MODEL_PREDICTION")

    def approve_entry(self, prediction: PredictionResult, min_probability: float = 0.60) -> bool:
        return bool(
            prediction.available
            and prediction.probability_target_before_stop is not None
            and prediction.probability_target_before_stop >= min_probability
            and (prediction.expected_r or -999.0) > 0.0
        )
