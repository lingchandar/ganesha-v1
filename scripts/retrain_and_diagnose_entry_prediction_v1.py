"""Retrain entry prediction v1, export OOS predictions, and print diagnostics."""
from __future__ import annotations

from pathlib import Path

from src.prediction.entry_prediction_model_v1 import (
    PRODUCTION_OOS_MIN_AUC,
    PRODUCTION_OOS_MIN_TOP_10_LIFT,
    PRODUCTION_OOS_MIN_TOP_5_LIFT,
    train,
)
from scripts.evaluate_entry_prediction_strategy_v1 import evaluate

DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
MODEL = Path("models/entry_prediction_v1.joblib")
PREDICTIONS = Path("data/prediction/entry_prediction_v1_oos_predictions.csv")


def main() -> None:
    print("=== RETRAIN ENTRY PREDICTION V1 ===")
    bundle = train(DATASET, MODEL)

    print("\n=== EXPORT OOS PREDICTIONS ===")
    report = evaluate(DATASET, PREDICTIONS)
    top_5_lift = report["top_k"]["top_5"]["daily_mean_lift"]
    top_10_lift = report["top_k"]["top_10"]["daily_mean_lift"]
    pooled_oos_ready = (
        report["auc"] >= PRODUCTION_OOS_MIN_AUC
        and top_5_lift >= PRODUCTION_OOS_MIN_TOP_5_LIFT
        and top_10_lift >= PRODUCTION_OOS_MIN_TOP_10_LIFT
    )

    print(f"oos_label_policy={report['label_policy']}")
    print(f"oos_rows={report['rows']}")
    print(f"oos_auc={report['auc']:.4f}")
    print(f"oos_ap={report['average_precision']:.4f}")
    print(f"oos_brier={report['brier']:.4f}")
    for key, metrics in report["top_k"].items():
        print(
            f"{key}=positive_rate={metrics['daily_mean_positive_rate']:.4f} "
            f"lift={metrics['daily_mean_lift']:.2f}x days={int(metrics['days_evaluated'])}"
        )
    print(f"fold_mean_production_ready={bundle['production_ready']}")
    print(f"pooled_oos_production_ready={pooled_oos_ready}")
    print(f"production_ready={pooled_oos_ready}")
    print(f"predictions_out={PREDICTIONS}")


if __name__ == "__main__":
    main()
