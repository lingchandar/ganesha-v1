from pathlib import Path

from src.prediction.momentum_ignition_model_v1 import DATASET_PATH, MODEL_PATH, train


if __name__ == "__main__":
    train(Path(DATASET_PATH), Path(MODEL_PATH))
