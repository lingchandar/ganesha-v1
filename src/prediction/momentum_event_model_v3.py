"""Leakage-safe walk-forward model for early momentum events.

v3 is designed around momentum_event_dataset_v2.csv. It predicts
``early_momentum_event_3d`` using only features known at the session close.
It evaluates ranked event capture (top 1/3/5/10 percent), uses chronological
outer folds with a purge gap, and calibrates probabilities with a second
chronological gap inside each training fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prediction.momentum_event_dataset_v2 import EVENT_HORIZON
from src.prediction.momentum_event_dataset import FEATURE_COLUMNS

LABEL_COLUMN = "early_momentum_event_3d"
MODEL_VERSION = "momentum_event_v3"
PURGE_SESSIONS = EVENT_HORIZON
CALIBRATION_GAP = EVENT_HORIZON
N_FOLDS = 4
VALIDATION_SESSIONS = 60
MIN_TRAIN_SESSIONS = 180
TOP_FRACTIONS = (0.01, 0.03, 0.05, 0.10)
MIN_MEAN_AUC = 0.60
MIN_MEAN_TOP5_LIFT = 1.75
MIN_MEAN_TOP10_LIFT = 1.40


@dataclass(frozen=True)
class FoldResult:
    model_name: str
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    train_rows: int
    validation_rows: int
    purge_rows: int
    validation_positive_rate: float
    auc: float
    average_precision: float
    brier: float
    top1_rate: float
    top1_lift: float
    top3_rate: float
    top3_lift: float
    top5_rate: float
    top5_lift: float
    top10_rate: float
    top10_lift: float


def _normal_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["trade_date"]).dt.normalize()


def _build_folds(dates: Iterable[pd.Timestamp], n_folds: int = N_FOLDS) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(pd.Series(list(dates)).dropna())))
    required = MIN_TRAIN_SESSIONS + n_folds * VALIDATION_SESSIONS
    if len(unique_dates) < required:
        raise ValueError(f"Not enough sessions: need {required}, got {len(unique_dates)}")
    first_validation = len(unique_dates) - n_folds * VALIDATION_SESSIONS
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        val_start = first_validation + fold * VALIDATION_SESSIONS
        val_end = val_start + VALIDATION_SESSIONS
        train_end = val_start - PURGE_SESSIONS
        if train_end >= MIN_TRAIN_SESSIONS:
            folds.append((fold + 1, np.arange(train_end), np.arange(val_start, val_end)))
    if len(folds) < 2:
        raise ValueError("Fewer than two walk-forward folds were produced")
    return folds


def _design_frame(df: pd.DataFrame, sector_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    numeric = df[FEATURE_COLUMNS].astype(float).copy()
    sector = pd.get_dummies(df["sector"].fillna("UNKNOWN").astype(str), prefix="sector", dtype=float)
    if sector_columns is None:
        sector_columns = sorted(sector.columns.tolist())
    sector = sector.reindex(columns=sector_columns, fill_value=0.0)
    design = pd.concat([numeric, sector], axis=1)
    return design, sector_columns


def _make_base_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=5.0,
            random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=35,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }


def _calibrated(model: object) -> CalibratedClassifierCV:
    cv = TimeSeriesSplit(n_splits=3, gap=CALIBRATION_GAP)
    return CalibratedClassifierCV(model, method="sigmoid", cv=cv)


def _validate_frame(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [LABEL_COLUMN, "ticker_symbol", "trade_date", "sector"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Momentum event v3 dataset contains NaN in required columns")
    if df.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Momentum event v3 dataset contains duplicate ticker/date rows")
    if not set(df[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} must be binary")


def _rank_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    y = y.reset_index(drop=True)
    base = float(y.mean())
    metrics: dict[str, float] = {}
    for fraction in TOP_FRACTIONS:
        key = int(fraction * 100)
        count = max(1, int(np.ceil(len(y) * fraction)))
        order = np.argsort(-probability)[:count]
        rate = float(y.iloc[order].mean())
        metrics[f"top{key}_rate"] = rate
        metrics[f"top{key}_lift"] = rate / base if base > 0 else float("nan")
    return metrics


def _production_ready(results: list[FoldResult]) -> bool:
    by_model: dict[str, list[FoldResult]] = {}
    for result in results:
        by_model.setdefault(result.model_name, []).append(result)
    for model_results in by_model.values():
        auc = float(np.mean([r.auc for r in model_results]))
        top5 = float(np.mean([r.top5_lift for r in model_results]))
        top10 = float(np.mean([r.top10_lift for r in model_results]))
        if auc >= MIN_MEAN_AUC and top5 >= MIN_MEAN_TOP5_LIFT and top10 >= MIN_MEAN_TOP10_LIFT:
            return True
    return False


def train_walk_forward(dataset_path: str, model_path: str) -> list[FoldResult]:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    df = df.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    _validate_frame(df)
    design, sector_columns = _design_frame(df)

    dates = _normal_dates(df)
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    date_position = {d: i for i, d in enumerate(unique_dates)}
    positions = dates.map(date_position).to_numpy()
    all_results: list[FoldResult] = []
    model_scores: dict[str, list[FoldResult]] = {}

    for model_name in _make_base_models():
        for fold, train_idx, val_idx in _build_folds(unique_dates):
            train_end = int(train_idx[-1])
            val_start = int(val_idx[0])
            val_end = int(val_idx[-1])
            train_mask = positions <= train_end
            val_mask = (positions >= val_start) & (positions <= val_end)
            train_df = df.loc[train_mask]
            val_df = df.loc[val_mask]
            if train_df["trade_date"].max() >= val_df["trade_date"].min():
                raise AssertionError("Training data reaches validation period")
            if train_df[LABEL_COLUMN].nunique() < 2 or val_df[LABEL_COLUMN].nunique() < 2:
                raise ValueError(f"Fold {fold} has insufficient class diversity")

            model = _calibrated(_make_base_models()[model_name])
            model.fit(design.loc[train_mask], train_df[LABEL_COLUMN])
            probability = model.predict_proba(design.loc[val_mask])[:, 1]
            metrics = _rank_metrics(val_df[LABEL_COLUMN], probability)
            result = FoldResult(
                model_name=model_name,
                fold=fold,
                train_start=str(train_df["trade_date"].min().date()),
                train_end=str(train_df["trade_date"].max().date()),
                validation_start=str(val_df["trade_date"].min().date()),
                validation_end=str(val_df["trade_date"].max().date()),
                train_rows=len(train_df),
                validation_rows=len(val_df),
                purge_rows=int(((positions > train_end) & (positions < val_start)).sum()),
                validation_positive_rate=float(val_df[LABEL_COLUMN].mean()),
                auc=float(roc_auc_score(val_df[LABEL_COLUMN], probability)),
                average_precision=float(average_precision_score(val_df[LABEL_COLUMN], probability)),
                brier=float(brier_score_loss(val_df[LABEL_COLUMN], probability)),
                top1_rate=metrics["top1_rate"], top1_lift=metrics["top1_lift"],
                top3_rate=metrics["top3_rate"], top3_lift=metrics["top3_lift"],
                top5_rate=metrics["top5_rate"], top5_lift=metrics["top5_lift"],
                top10_rate=metrics["top10_rate"], top10_lift=metrics["top10_lift"],
            )
            all_results.append(result)
            model_scores.setdefault(model_name, []).append(result)
            print(
                f"model={model_name} fold={fold} train={result.train_start}:{result.train_end} "
                f"validation={result.validation_start}:{result.validation_end} "
                f"auc={result.auc:.4f} ap={result.average_precision:.4f} brier={result.brier:.4f} "
                f"top1={result.top1_rate:.4f}/{result.top1_lift:.2f} "
                f"top3={result.top3_rate:.4f}/{result.top3_lift:.2f} "
                f"top5={result.top5_rate:.4f}/{result.top5_lift:.2f} "
                f"top10={result.top10_rate:.4f}/{result.top10_lift:.2f}"
            )

    ranked = sorted(
        model_scores.items(),
        key=lambda item: (
            float(np.mean([r.top5_lift for r in item[1]])),
            float(np.mean([r.auc for r in item[1]])),
        ),
        reverse=True,
    )
    champion_name = ranked[0][0]
    champion = _calibrated(_make_base_models()[champion_name])
    champion.fit(design, df[LABEL_COLUMN])
    ready = _production_ready(all_results)

    artifact = {
        "model_version": MODEL_VERSION,
        "model": champion,
        "model_name": champion_name,
        "production_ready": ready,
        "feature_columns": list(design.columns),
        "sector_columns": sector_columns,
        "label_column": LABEL_COLUMN,
        "event_horizon_sessions": EVENT_HORIZON,
        "purge_sessions": PURGE_SESSIONS,
        "calibration_gap": CALIBRATION_GAP,
        "top_fractions": TOP_FRACTIONS,
        "production_gate": {
            "min_mean_auc": MIN_MEAN_AUC,
            "min_mean_top5_lift": MIN_MEAN_TOP5_LIFT,
            "min_mean_top10_lift": MIN_MEAN_TOP10_LIFT,
        },
        "walk_forward_folds": [r.__dict__ for r in all_results],
        "dataset_start": str(df["trade_date"].min().date()),
        "dataset_end": str(df["trade_date"].max().date()),
        "dataset_rows": len(df),
        "event_rate": float(df[LABEL_COLUMN].mean()),
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)

    print(f"model_version={MODEL_VERSION}")
    for name, results in model_scores.items():
        print(
            f"summary={name} mean_auc={np.mean([r.auc for r in results]):.4f} "
            f"mean_ap={np.mean([r.average_precision for r in results]):.4f} "
            f"mean_brier={np.mean([r.brier for r in results]):.4f} "
            f"mean_top5_lift={np.mean([r.top5_lift for r in results]):.2f} "
            f"mean_top10_lift={np.mean([r.top10_lift for r in results]):.2f}"
        )
    print(f"champion={champion_name}")
    print(f"production_ready={ready}")
    print(f"saved_model={path}")
    return all_results
