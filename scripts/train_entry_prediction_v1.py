from pathlib import Path

from src.prediction.entry_prediction_model_v1 import train


if __name__ == "__main__":
    train(
        Path("data/prediction/entry_prediction_dataset_v1.csv"),
        Path("models/entry_prediction_v1.joblib"),
    )
