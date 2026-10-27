"""Point-in-time trade prediction engine for Ganesha v1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

MODEL_VERSION = "entry_v4"
DEFAULT_MODEL_PATH = Path("models/entry_prediction_v4.joblib")

SETUP_TYPES = (
    "CLASSIC_RESISTANCE_BREAKOUT",
    "BREAKOUT_RETEST",
    "TREND_PULLBACK",
    "DEMAND_SUPPORT_BOUNCE",
    "TREND_CONTINUATION_FLAG",
    "RS_LEADER_BREAKOUT",
)

REGIME_FEATURE_COLUMNS = [
    "trend_bullish",
    "trend_bearish",
    "momentum_positive",
    "momentum_negative",
    "volatility_low",
    "volatility_high",
]

REGIME_INTERACTION_FEATURE_COLUMNS = [
    "return_5d_x_trend_bullish",
    "return_5d_x_trend_bearish",
    "rsi_14_x_momentum_positive",
    "rsi_14_x_momentum_negative",
    "atr_pct_x_trend_bullish",
    "atr_pct_x_trend_bearish",
]


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic regime and interaction features from point-in-time inputs."""
    result = frame.copy()
    trend_bullish = (
        (result["price_vs_ema200_pct"] > 0)
        & (result["ema20_vs_ema50_pct"] > 0)
        & (result["ema50_vs_ema200_pct"] > 0)
    )
    trend_bearish = (
        (result["price_vs_ema200_pct"] < 0)
        & (result["ema20_vs_ema50_pct"] < 0)
        & (result["ema50_vs_ema200_pct"] < 0)
    )
    momentum_positive = (
        (result["return_5d"] > 0)
        & (result["plus_di_minus_di"] > 0)
        & (result["rsi_14"] >= 55)
    )
    momentum_negative = (
        (result["return_5d"] < 0)
        & (result["plus_di_minus_di"] < 0)
        & (result["rsi_14"] <= 45)
    )

    # Volatility interactions are based on within-frame terciles. During training
    # the full historical frame is used only to construct a deterministic feature;
    # the production model does not use a separately trained volatility model.
    low_cut, high_cut = result["atr_pct"].quantile([1 / 3, 2 / 3])
    volatility_low = result["atr_pct"] <= low_cut
    volatility_high = result["atr_pct"] > high_cut

    result["trend_bullish"] = trend_bullish.astype(float)
    result["trend_bearish"] = trend_bearish.astype(float)
    result["momentum_positive"] = momentum_positive.astype(float)
    result["momentum_negative"] = momentum_negative.astype(float)
    result["volatility_low"] = volatility_low.astype(float)
    result["volatility_high"] = volatility_high.astype(float)
    result["return_5d_x_trend_bullish"] = result["return_5d"] * result["trend_bullish"]
    result["return_5d_x_trend_bearish"] = result["return_5d"] * result["trend_bearish"]
    result["rsi_14_x_momentum_positive"] = result["rsi_14"] * result["momentum_positive"]
    result["rsi_14_x_momentum_negative"] = result["rsi_14"] * result["momentum_negative"]
    result["atr_pct_x_trend_bullish"] = result["atr_pct"] * result["trend_bullish"]
    result["atr_pct_x_trend_bearish"] = result["atr_pct"] * result["trend_bearish"]
    return result


@dataclass(frozen=True)
class PredictionResult:
    available: bool
    probability_target_before_stop: Optional[float]
    expected_r: Optional[float]
    confidence: Optional[float]
    model_version: Optional[str]
    reason: str


def _signal_value(signal: Any, name: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(name, default)
    return getattr(signal, name, default)


class PredictionFeatureBuilder:
    """Build features using only data known at the signal timestamp."""

    NUMERIC_FEATURE_COLUMNS = [
        "rsi_14", "rsi_change_5d", "atr_pct", "atr_expansion_ratio",
        "volume_surge_multiple", "risk_reward_ratio", "distance_to_stop_atr",
        "distance_to_target_atr", "price_vs_ema20_pct", "price_vs_ema50_pct",
        "price_vs_ema200_pct", "ema20_vs_ema50_pct", "ema50_vs_ema200_pct",
        "ema50_slope_20", "return_5d", "return_20d", "volatility_20d",
        "adx_14", "plus_di_minus_di", "macd_hist_pct", "macd_hist_diff_pct",
    ]
    SETUP_FEATURE_COLUMNS = [f"setup_{setup.lower()}" for setup in SETUP_TYPES]
    FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + SETUP_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS + REGIME_INTERACTION_FEATURE_COLUMNS

    @classmethod
    def build(cls, signal: Any, df_features: pd.DataFrame) -> pd.DataFrame:
        if df_features.empty:
            raise ValueError("Feature frame cannot be empty")
        latest = df_features.iloc[-1]
        close = max(float(latest["close_price"]), 1e-9)
        atr = max(float(latest.get("atr_14", 0.0)), 1e-9)
        ema20 = float(latest.get("ema_20", close)); ema50 = float(latest.get("ema_50", close)); ema200 = float(latest.get("ema_200", close))
        entry = float(_signal_value(signal, "entry_price", close)); stop = float(_signal_value(signal, "stop_loss", entry)); target = float(_signal_value(signal, "target_price", entry))
        risk = max(entry - stop, 0.0); reward = max(target - entry, 0.0)
        close_series = pd.to_numeric(df_features["close_price"], errors="coerce")
        ret5 = float(close_series.pct_change(5).iloc[-1]) if len(close_series) > 5 else 0.0
        ret20 = float(close_series.pct_change(20).iloc[-1]) if len(close_series) > 20 else 0.0
        vol20 = float(close_series.pct_change().rolling(20).std().iloc[-1]) if len(close_series) > 20 else 0.0
        rsi_series = pd.to_numeric(df_features.get("rsi_14", pd.Series(dtype=float)), errors="coerce")
        rsi_change_5d = float(rsi_series.iloc[-1] - rsi_series.iloc[-6]) if len(rsi_series) > 5 else 0.0
        atr_sma20 = float(latest.get("atr_sma_20", atr)); atr_expansion_ratio = atr / max(atr_sma20, 1e-9)
        macd_hist = float(latest.get("macd_hist", 0.0)); macd_hist_diff = float(latest.get("macd_hist_diff", 0.0))
        adx = float(latest.get("adx_14", 0.0)); plus_di = float(latest.get("plus_di", 0.0)); minus_di = float(latest.get("minus_di", 0.0))
        setup_type = str(_signal_value(signal, "setup_type", ""))
        if setup_type not in SETUP_TYPES:
            raise ValueError(f"Unsupported setup type for prediction: {setup_type}")
        row = {
            "rsi_14": float(latest.get("rsi_14", 50.0)), "rsi_change_5d": rsi_change_5d,
            "atr_pct": atr / close, "atr_expansion_ratio": atr_expansion_ratio,
            "volume_surge_multiple": float(_signal_value(signal, "volume_surge_multiple", 1.0)),
            "risk_reward_ratio": reward / risk if risk > 0 else 0.0, "distance_to_stop_atr": risk / atr,
            "distance_to_target_atr": reward / atr, "price_vs_ema20_pct": (close / ema20) - 1.0 if ema20 else 0.0,
            "price_vs_ema50_pct": (close / ema50) - 1.0 if ema50 else 0.0, "price_vs_ema200_pct": (close / ema200) - 1.0 if ema200 else 0.0,
            "ema20_vs_ema50_pct": (ema20 / ema50) - 1.0 if ema50 else 0.0, "ema50_vs_ema200_pct": (ema50 / ema200) - 1.0 if ema200 else 0.0,
            "ema50_slope_20": float(latest.get("ema_50_slope_20", 0.0)), "return_5d": ret5, "return_20d": ret20,
            "volatility_20d": vol20, "adx_14": adx, "plus_di_minus_di": plus_di - minus_di,
            "macd_hist_pct": macd_hist / close, "macd_hist_diff_pct": macd_hist_diff / close,
        }
        row.update({column: float(column == f"setup_{setup_type.lower()}") for column in cls.SETUP_FEATURE_COLUMNS})
        base = pd.DataFrame([row], columns=cls.NUMERIC_FEATURE_COLUMNS + cls.SETUP_FEATURE_COLUMNS)
        base = add_regime_features(base)
        return base[cls.FEATURE_COLUMNS]


class EntryPredictionEngine:
    """Loads a trained, temporally validated model for entry decisions."""
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path); self._bundle: Optional[Dict] = None
        if self.model_path.exists(): self._bundle = joblib.load(self.model_path); self._validate_bundle()
    @property
    def available(self) -> bool: return self._bundle is not None
    def _validate_bundle(self) -> None:
        if not isinstance(self._bundle, dict): raise ValueError("Prediction model bundle must be a dictionary")
        if self._bundle.get("model_version") != MODEL_VERSION: raise ValueError("Unsupported prediction model version")
        if not self._bundle.get("feature_columns"): raise ValueError("Prediction model bundle has no feature columns")
        if "model" not in self._bundle: raise ValueError("Prediction model bundle has no model")
    def predict(self, signal: Any, df_features: pd.DataFrame) -> PredictionResult:
        if not self.available: return PredictionResult(False, None, None, None, None, "MODEL_NOT_TRAINED")
        features = PredictionFeatureBuilder.build(signal, df_features).reindex(columns=self._bundle["feature_columns"], fill_value=0.0)
        probability = float(self._bundle["model"].predict_proba(features)[0, 1]); expected_r = (probability * 2.0) + ((1.0 - probability) * -1.0); confidence = abs(probability - 0.5) * 2.0
        return PredictionResult(True, probability, expected_r, confidence, MODEL_VERSION, "MODEL_PREDICTION")
    def approve_entry(self, prediction: PredictionResult, min_probability: float = 0.60) -> bool:
        return bool(prediction.available and prediction.probability_target_before_stop is not None and prediction.probability_target_before_stop >= min_probability and (prediction.expected_r or -999.0) > 0.0)
