"""Score completed daily bars for next-session momentum ignition."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/momentum_ignition_v1.joblib")
DEFAULT_DATASET_PATH = Path("data/prediction/momentum_ignition_dataset_v3.csv")


class MomentumIgnitionScorer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.features = list(artifact["features"])
        self.label = artifact["label"]
        self.model_version = artifact["model_version"]

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.features if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing scoring features: {missing}")

        scored = frame.copy()
        features = scored[self.features].apply(pd.to_numeric, errors="coerce")
        probabilities = np.asarray(self.model.predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Model predict_proba must return a 2-column probability matrix")
        scored["ignition_probability"] = probabilities[:, 1]
        scored["model_version"] = self.model_version
        return scored


def score_latest(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    latest_date = frame["trade_date"].max()
    latest = frame[frame["trade_date"] == latest_date].copy()
    scorer = MomentumIgnitionScorer(model_path)
    scored = scorer.score(latest)
    return scored.sort_values("ignition_probability", ascending=False).reset_index(drop=True)
