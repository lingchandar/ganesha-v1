"""Evaluate the entry prediction model on its untouched temporal validation set."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATASET = Path("data/prediction/entry_dataset.csv")
MODEL = Path("models/entry_prediction_v1.joblib")
LABEL = "label_target_before_stop"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main() -> None:
    bundle = joblib.load(MODEL)
    df = pd.read_csv(DATASET, parse_dates=["trade_date"])
    validation_start = pd.Timestamp(bundle["validation_start"])
    validation = df[df["trade_date"] >= validation_start].copy()
    validation = validation.sort_values(["trade_date", "ticker_symbol", "setup_type"])

    features = bundle["feature_columns"]
    probabilities = bundle["model"].predict_proba(validation[features])[:, 1]
    validation["prediction_probability"] = probabilities
    y = validation[LABEL].astype(int)

    print("=== ENTRY PREDICTION VALIDATION ===")
    print(f"validation_rows={len(validation)}")
    print(f"validation_dates={validation['trade_date'].min().date()} to {validation['trade_date'].max().date()}")
    print(f"positive_rate={y.mean():.4f}")
    print(f"auc={roc_auc_score(y, probabilities):.4f}")
    print(f"average_precision={average_precision_score(y, probabilities):.4f}")
    print(f"brier_score={brier_score_loss(y, probabilities):.4f}")

    print("\n=== THRESHOLD ANALYSIS ===")
    print("threshold signals coverage success_rate avg_R total_R precision")
    for threshold in THRESHOLDS:
        accepted = validation[validation["prediction_probability"] >= threshold]
        if accepted.empty:
            print(f"{threshold:.2f} 0 0.0000 0.0000 0.0000 0.0000 0.0000")
            continue
        success_rate = accepted[LABEL].mean()
        avg_r = (success_rate * 2.0) + ((1.0 - success_rate) * -1.0)
        total_r = avg_r * len(accepted)
        coverage = len(accepted) / len(validation)
        print(f"{threshold:.2f} {len(accepted)} {coverage:.4f} {success_rate:.4f} {avg_r:.4f} {total_r:.2f} {success_rate:.4f}")

    print("\n=== SETUP PERFORMANCE AT THRESHOLD 0.60 ===")
    accepted = validation[validation["prediction_probability"] >= 0.60]
    if accepted.empty:
        print("No validation signals passed threshold 0.60")
    else:
        setup_stats = (
            accepted.groupby("setup_type")[LABEL]
            .agg(["count", "mean"])
            .rename(columns={"count": "signals", "mean": "success_rate"})
            .sort_values("success_rate", ascending=False)
        )
        setup_stats["avg_R"] = setup_stats["success_rate"] * 3.0 - 1.0
        print(setup_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== TOP PREDICTIONS ===")
    print(validation[["trade_date", "ticker_symbol", "setup_type", LABEL, "prediction_probability"]].sort_values("prediction_probability", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
