from __future__ import annotations

import argparse

from src.prediction.momentum_event_model_v2 import train_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark momentum event model v2")
    parser.add_argument("--dataset", default="data/prediction/momentum_event_dataset_v1.csv")
    parser.add_argument("--model", default="models/momentum_event_v2.joblib")
    args = parser.parse_args()
    train_walk_forward(args.dataset, args.model)


if __name__ == "__main__":
    main()
