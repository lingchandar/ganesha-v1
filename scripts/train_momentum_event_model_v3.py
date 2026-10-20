"""Train and report the leakage-safe momentum event model v3."""
from __future__ import annotations

from src.prediction.momentum_event_model_v3 import train_walk_forward

DATASET = "data/prediction/momentum_event_dataset_v2.csv"
MODEL = "models/momentum_event_v3.joblib"


if __name__ == "__main__":
    train_walk_forward(DATASET, MODEL)
