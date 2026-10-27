"""Read-only shadow inference for the Extra Trees candidate bundle.

This module deliberately has no EntryGate integration and no approval logic.
It only observes signals that the live pipeline already produced and records
what the candidate model would have predicted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from src.prediction.engine import PredictionFeatureBuilder

CANDIDATE_VERSION = "entry_prediction_extra_trees_candidate_v1"
DEFAULT_MODEL_PATH = Path("models/entry_prediction_extra_trees_candidate_v1.joblib")


@dataclass(frozen=True)
class ShadowPrediction:
    available: bool
    symbol: str
    setup_type: str
    probability_target_before_stop: Optional[float]
    confidence: Optional[float]
    model_version: Optional[str]
    reason: str


class CandidateShadowPredictor:
    """Load the candidate and produce observation-only predictions."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._bundle: Optional[dict[str, Any]] = None
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self._validate_bundle(bundle)
            self._bundle = bundle

    @staticmethod
    def _validate_bundle(bundle: dict[str, Any]) -> None:
        if bundle.get("model_version") != CANDIDATE_VERSION:
            raise ValueError("Unsupported candidate model version")
        if bundle.get("algorithm") != "extra_trees":
            raise ValueError("Shadow bundle must use Extra Trees")
        if bundle.get("production_ready") is not False:
            raise ValueError("Shadow bundle must remain production_ready=False")
        if bundle.get("promotion_status") != "candidate_only":
            raise ValueError("Shadow bundle must remain candidate_only")
        if list(bundle.get("feature_columns", [])) != list(PredictionFeatureBuilder.FEATURE_COLUMNS):
            raise ValueError("Candidate feature schema does not match live feature builder")
        if "model" not in bundle:
            raise ValueError("Candidate bundle has no model")

    @property
    def available(self) -> bool:
        return self._bundle is not None

    def predict(self, signal: Any, df_features: pd.DataFrame) -> ShadowPrediction:
        symbol = str(getattr(signal, "ticker_symbol", ""))
        setup_type = str(getattr(signal, "setup_type", ""))
        if not self.available:
            return ShadowPrediction(False, symbol, setup_type, None, None, None, "CANDIDATE_NOT_TRAINED")

        features = PredictionFeatureBuilder.build(signal, df_features).reindex(
            columns=self._bundle["feature_columns"], fill_value=0.0
        )
        probability = float(self._bundle["model"].predict_proba(features)[0, 1])
        confidence = abs(probability - 0.5) * 2.0
        return ShadowPrediction(
            True,
            symbol,
            setup_type,
            probability,
            confidence,
            CANDIDATE_VERSION,
            "SHADOW_PREDICTION",
        )
