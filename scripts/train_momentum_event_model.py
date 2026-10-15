"""Train and evaluate the momentum-event model with purged walk-forward validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.prediction.momentum_event_model import train_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/prediction/momentum_event_dataset_v1.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/momentum_event_v1.joblib"),
    )
    args = parser.parse_args()
    train_walk_forward(str(args.dataset), str(args.model))


if __name__ == "__main__":
    main()
