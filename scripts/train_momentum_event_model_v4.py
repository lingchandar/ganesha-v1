"""Train and report Momentum Event Model v4."""
from __future__ import annotations

from src.prediction.momentum_event_model_v4 import train_walk_forward

DATASET = "data/prediction/momentum_event_dataset_v2.csv"
MODEL = "models/momentum_event_v4.joblib"

if __name__ == "__main__":
    train_walk_forward(DATASET, MODEL)
