"""Retrain entry prediction v1, export OOS predictions, and print diagnostics."""
from __future__ import annotations

from pathlib import Path

from src.prediction.entry_prediction_model_v1 import train
from scripts.evaluate_entry_prediction_strategy_v1 import evaluate

DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
MODEL = Path("models/entry_prediction_v1.joblib")
PREDICTIONS = Path("data/prediction/entry_prediction_v1_oos_predictions.csv")


def main() -> None:
    print("=== RETRAIN ENTRY PREDICTION V1 ===")
    bundle = train(DATASET, MODEL)

    print("\n=== EXPORT OOS PREDICTIONS ===")
    report = evaluate(DATASET, PREDICTIONS)
    print(f"oos_auc={report['auc']:.4f}")
    print(f"oos_ap={report['average_precision']:.4f}")
    print(f"oos_brier={report['brier']:.4f}")
    print(f"production_ready={bundle['production_ready']}")
    print(f"predictions_out={PREDICTIONS}")


if __name__ == "__main__":
    main()
