"""Quick diagnostic audit for Momentum Event Model v4.

This is an exploratory diagnostic, not an out-of-sample validation report. The v4
walk-forward metrics printed by the training script are the correct evidence for
production readiness. This script answers two quick questions:
1. What features does the saved champion model rely on?
2. Which stocks are ranked highest on the latest completed session?
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.prediction.momentum_event_model_v4 import _design_frame

DATASET = Path("data/prediction/momentum_event_dataset_v2.csv")
MODEL = Path("models/momentum_event_v4.joblib")
TOP_N = 15


def _feature_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from the saved champion when available."""
    estimators = getattr(model, "calibrated_classifiers_", None)
    rows: list[tuple[str, float]] = []
    if estimators:
        for calibrated in estimators:
            estimator = getattr(calibrated, "estimator", None)
            importance = getattr(estimator, "feature_importances_", None)
            if importance is not None:
                rows.append(("__fold__", float(np.mean(importance))))
                values = np.asarray(importance, dtype=float)
                if len(values) == len(feature_names):
                    rows = [(name, float(value)) for name, value in zip(feature_names, values)]
                    break
    if not rows:
        estimator = getattr(model, "estimator", None)
        importance = getattr(estimator, "feature_importances_", None)
        if importance is not None and len(importance) == len(feature_names):
            rows = [(name, float(value)) for name, value in zip(feature_names, importance)]
    if not rows:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(rows, columns=["feature", "importance"]).sort_values("importance", ascending=False)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}. Run train_momentum_event_model_v4.py first.")

    artifact = joblib.load(MODEL)
    df = pd.read_csv(DATASET, parse_dates=["trade_date"])
    df = df.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    design, _ = _design_frame(df, artifact.get("sector_columns"))
    model = artifact["model"]
    probability = model.predict_proba(design)[:, 1]
    df["model_probability"] = probability

    latest_date = df["trade_date"].max()
    latest = df[df["trade_date"] == latest_date].copy().sort_values("model_probability", ascending=False)

    print(f"model_version={artifact.get('model_version')}")
    print(f"champion={artifact.get('model_name')}")
    print(f"production_ready={artifact.get('production_ready')}")
    print(f"latest_completed_session={latest_date.date()}")
    print(f"latest_rows={len(latest)}")
    print("NOTE=in-sample diagnostic only; do not treat latest probabilities as OOS accuracy")

    print("\nTOP_LATEST_RANKED_STOCKS")
    columns = [
        "ticker_symbol", "model_probability", "return_1d", "return_3d", "return_5d",
        "volume_ratio_20d", "volume_zscore_20d", "atr_expansion_ratio",
        "relative_strength_20d", "sector_strength_20d",
        "distance_to_20d_high_pct", "distance_to_50d_high_pct",
    ]
    available = [column for column in columns if column in latest.columns]
    print(latest[available].head(TOP_N).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    importance = _feature_importance(model, list(design.columns))
    if not importance.empty:
        print("\nTOP_FEATURE_IMPORTANCE")
        print(importance.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    else:
        print("\nTOP_FEATURE_IMPORTANCE\nnot_available_for_saved_model")


if __name__ == "__main__":
    main()
