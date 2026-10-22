"""Leakage-safe walk-forward model for next-session momentum ignition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("data/prediction/momentum_ignition_dataset_v3.csv")
MODEL_PATH = Path("models/momentum_ignition_v1.joblib")
LABEL = "ignition_1d"
PURGE_SESSIONS = 1
VALIDATION_SESSIONS = 60
MIN_TRAIN_SESSIONS = 180

FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_acceleration_1d",
    "return_acceleration_3d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "range_pct",
    "range_atr_multiple",
    "atr_pct",
    "atr_expansion_ratio",
    "gap_pct",
    "rsi_14",
    "rsi_change_3d",
    "macd_hist_pct",
    "macd_hist_change_pct",
    "adx_14",
    "plus_di_minus_di",
    "price_vs_ema20_pct",
    "price_vs_ema50_pct",
    "price_vs_ema200_pct",
    "ema20_vs_ema50_pct",
    "ema50_vs_ema200_pct",
    "ema50_slope_20",
    "distance_to_20d_high_pct",
    "distance_to_50d_high_pct",
    "range_compression_10d",
    "relative_strength_5d",
    "relative_strength_20d",
    "sector_strength_5d",
    "sector_strength_20d",
    "delivery_pct",
    "delivery_pct_ratio_20d",
    "delivery_available",
]


def _make_model() -> Pipeline:
    estimator = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        max_iter=1500,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=3,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", calibrated),
        ]
    )


@dataclass(frozen=True)
class FoldResult:
    validation_start: str
    validation_end: str
    auc: float
    average_precision: float
    top5_lift: float
    top10_lift: float
    signals_top5: int
    signals_top10: int


def _rank_lift(y: np.ndarray, probabilities: np.ndarray, fraction: float) -> tuple[float, int]:
    n = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(probabilities)[::-1][:n]
    selected_rate = float(y[order].mean())
    baseline = float(y.mean())
    return (selected_rate / baseline if baseline > 0 else 0.0), n


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    return frame[FEATURES].apply(pd.to_numeric, errors="coerce")


def walk_forward_validate(frame: pd.DataFrame) -> list[FoldResult]:
    frame = frame.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    dates = sorted(frame["trade_date"].dropna().unique())
    results: list[FoldResult] = []

    for end in range(MIN_TRAIN_SESSIONS, len(dates), VALIDATION_SESSIONS):
        train_end_index = end - PURGE_SESSIONS
        if train_end_index <= 0:
            continue
        validation_dates = dates[end : end + VALIDATION_SESSIONS]
        if len(validation_dates) < VALIDATION_SESSIONS:
            break
        train_dates = dates[:train_end_index]
        train = frame[frame["trade_date"].isin(train_dates)]
        validation = frame[frame["trade_date"].isin(validation_dates)]
        if train[LABEL].nunique() < 2 or validation[LABEL].nunique() < 2:
            continue

        model = _make_model()
        x_train = _build_features(train)
        y_train = train[LABEL].astype(int).to_numpy()
        x_valid = _build_features(validation)
        y_valid = validation[LABEL].astype(int).to_numpy()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]

        top5, n5 = _rank_lift(y_valid, probabilities, 0.05)
        top10, n10 = _rank_lift(y_valid, probabilities, 0.10)
        results.append(
            FoldResult(
                str(pd.Timestamp(validation_dates[0]).date()),
                str(pd.Timestamp(validation_dates[-1]).date()),
                float(roc_auc_score(y_valid, probabilities)),
                float(average_precision_score(y_valid, probabilities)),
                top5,
                top10,
                n5,
                n10,
            )
        )
    return results


def train(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> None:
    frame = pd.read_csv(dataset_path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", LABEL]).copy()

    results = walk_forward_validate(frame)
    if not results:
        raise RuntimeError("No valid walk-forward folds were produced")

    mean_auc = float(np.mean([r.auc for r in results]))
    mean_ap = float(np.mean([r.average_precision for r in results]))
    mean_top5 = float(np.mean([r.top5_lift for r in results]))
    mean_top10 = float(np.mean([r.top10_lift for r in results]))
    production_ready = mean_auc >= 0.60 and mean_top5 >= 1.75 and mean_top10 >= 1.40

    print(f"model=logistic_regression mean_auc={mean_auc:.4f} mean_ap={mean_ap:.4f} mean_top5_lift={mean_top5:.2f} mean_top10_lift={mean_top10:.2f}")
    for index, result in enumerate(results, start=1):
        print(
            f"fold={index} validation={result.validation_start}..{result.validation_end} "
            f"auc={result.auc:.4f} ap={result.average_precision:.4f} "
            f"top5_lift={result.top5_lift:.2f} top10_lift={result.top10_lift:.2f}"
        )
    print(f"production_ready={production_ready}")

    if not production_ready:
        print("model_not_saved_for_production=walk-forward gate failed")
        return

    final_model = _make_model()
    final_model.fit(_build_features(frame), frame[LABEL].astype(int))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "label": LABEL,
            "model_version": "momentum_ignition_v1",
        },
        model_path,
    )
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    train()
