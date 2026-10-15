"""Walk-forward benchmark for the Ganesha early momentum-event model.

v2 deliberately treats the existing v1 model as a baseline. It benchmarks two
model families, keeps sector information as a categorical feature, uses
chronological folds with a purge gap, and refuses to mark a model production
ready when out-of-sample evidence is weak.
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.momentum_event_dataset import FEATURE_COLUMNS, HORIZON_5

LABEL_COLUMN = "hit_up_before_down_5d"
MODEL_VERSION = "momentum_event_v2"
PURGE_SESSIONS = HORIZON_5
N_FOLDS = 4
MIN_TRAIN_SESSIONS = 180
VALIDATION_SESSIONS = 60
MIN_PRODUCTION_AUC = 0.60
MIN_PRODUCTION_TOP10_LIFT = 1.30


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
    purged_rows: int
    validation_positive_rate: float
    auc: float
    average_precision: float
    brier: float
    top_decile_rate: float
    top_decile_lift: float


def _normal_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["trade_date"]).dt.normalize()


def _build_folds(dates: Iterable[pd.Timestamp], n_folds: int = N_FOLDS) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(pd.Series(list(dates)).dropna())))
    if len(unique_dates) < MIN_TRAIN_SESSIONS + n_folds * VALIDATION_SESSIONS:
        raise ValueError("Not enough trading sessions for requested walk-forward folds")
    first_validation = len(unique_dates) - n_folds * VALIDATION_SESSIONS
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        val_start = first_validation + fold * VALIDATION_SESSIONS
        val_end = val_start + VALIDATION_SESSIONS
        train_end = val_start - PURGE_SESSIONS
        if train_end < MIN_TRAIN_SESSIONS:
            continue
        folds.append((fold + 1, np.arange(0, train_end), np.arange(val_start, val_end)))
    if len(folds) < 2:
        raise ValueError("Walk-forward configuration produced fewer than two folds")
    return folds


def _sector_columns(df: pd.DataFrame) -> list[str]:
    sectors = pd.get_dummies(df["sector"].fillna("UNKNOWN").astype(str), prefix="sector", dtype=float)
    return list(sectors.columns)


def _design_frame(df: pd.DataFrame, sector_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    numeric = df[FEATURE_COLUMNS].astype(float).copy()
    sector = pd.get_dummies(df["sector"].fillna("UNKNOWN").astype(str), prefix="sector", dtype=float)
    if sector_columns is None:
        sector_columns = list(sector.columns)
    sector = sector.reindex(columns=sector_columns, fill_value=0.0)
    return pd.concat([numeric, sector], axis=1), sector_columns


def _make_models() -> dict[str, CalibratedClassifierCV]:
    hist = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=3.0,
        random_state=42,
    )
    extra = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    cv = TimeSeriesSplit(n_splits=3)
    return {
        "hist_gradient_boosting": CalibratedClassifierCV(hist, method="sigmoid", cv=cv),
        "extra_trees": CalibratedClassifierCV(extra, method="sigmoid", cv=cv),
    }


def _validate_frame(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [LABEL_COLUMN, "ticker_symbol", "trade_date", "sector"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Momentum event model dataset contains NaN in required columns")
    if df.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Momentum event model dataset contains duplicate ticker/date rows")
    if not set(df[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} must be binary")


def _top_decile_metrics(y: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    count = max(1, int(np.ceil(len(y) * 0.10)))
    order = np.argsort(-probability)[:count]
    top_rate = float(y.iloc[order].mean())
    base_rate = float(y.mean())
    lift = top_rate / base_rate if base_rate > 0 else float("nan")
    return top_rate, lift


def _production_ready(results: list[FoldResult]) -> bool:
    by_model: dict[str, list[FoldResult]] = {}
    for result in results:
        by_model.setdefault(result.model_name, []).append(result)
    for model_results in by_model.values():
        mean_auc = float(np.mean([r.auc for r in model_results]))
        mean_lift = float(np.mean([r.top_decile_lift for r in model_results]))
        if mean_auc >= MIN_PRODUCTION_AUC and mean_lift >= MIN_PRODUCTION_TOP10_LIFT:
            return True
    return False


def train_walk_forward(dataset_path: str, model_path: str) -> list[FoldResult]:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    df = df.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    _validate_frame(df)
    design, sector_columns = _design_frame(df)

    dates = _normal_dates(df)
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    date_to_position = {value: pos for pos, value in enumerate(unique_dates)}
    date_positions = dates.map(date_to_position).to_numpy()
    results: list[FoldResult] = []
    model_scores: dict[str, list[FoldResult]] = {}

    for model_name in _make_models():
        for fold, train_date_idx, val_date_idx in _build_folds(unique_dates):
            val_start_pos = int(val_date_idx[0])
            val_end_pos = int(val_date_idx[-1])
            train_end_pos = int(train_date_idx[-1])
            train_mask = date_positions <= train_end_pos
            val_mask = (date_positions >= val_start_pos) & (date_positions <= val_end_pos)
            train_df = df.loc[train_mask]
            val_df = df.loc[val_mask]
            X_train = design.loc[train_mask]
            X_val = design.loc[val_mask]
            if train_df["trade_date"].max() >= val_df["trade_date"].min():
                raise AssertionError("Training data reaches validation period")
            if train_df[LABEL_COLUMN].nunique() < 2 or val_df[LABEL_COLUMN].nunique() < 2:
                raise ValueError(f"Fold {fold} has insufficient class diversity")

            model = _make_models()[model_name]
            model.fit(X_train, train_df[LABEL_COLUMN])
            probability = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(val_df[LABEL_COLUMN], probability)
            ap = average_precision_score(val_df[LABEL_COLUMN], probability)
            brier = brier_score_loss(val_df[LABEL_COLUMN], probability)
            top_rate, lift = _top_decile_metrics(val_df[LABEL_COLUMN].reset_index(drop=True), probability)
            result = FoldResult(
                model_name=model_name,
                fold=fold,
                train_start=str(train_df["trade_date"].min().date()),
                train_end=str(train_df["trade_date"].max().date()),
                validation_start=str(val_df["trade_date"].min().date()),
                validation_end=str(val_df["trade_date"].max().date()),
                train_rows=len(train_df),
                validation_rows=len(val_df),
                purged_rows=int((~train_mask & ~val_mask).sum()),
                validation_positive_rate=float(val_df[LABEL_COLUMN].mean()),
                auc=float(auc),
                average_precision=float(ap),
                brier=float(brier),
                top_decile_rate=float(top_rate),
                top_decile_lift=float(lift),
            )
            results.append(result)
            model_scores.setdefault(model_name, []).append(result)
            print(
                f"model={model_name} fold={fold} train={result.train_start}:{result.train_end} "
                f"validation={result.validation_start}:{result.validation_end} "
                f"auc={result.auc:.4f} ap={result.average_precision:.4f} "
                f"brier={result.brier:.4f} top10_rate={result.top_decile_rate:.4f} "
                f"top10_lift={result.top_decile_lift:.2f}"
            )

    ranked = sorted(
        model_scores.items(),
        key=lambda item: (
            float(np.mean([r.auc for r in item[1]])),
            float(np.mean([r.top_decile_lift for r in item[1]])),
        ),
        reverse=True,
    )
    champion_name = ranked[0][0]
    champion = _make_models()[champion_name]
    champion.fit(design, df[LABEL_COLUMN])
    ready = _production_ready(results)

    output = {
        "model_version": MODEL_VERSION,
        "model": champion,
        "model_name": champion_name,
        "production_ready": ready,
        "production_gate": {
            "min_mean_auc": MIN_PRODUCTION_AUC,
            "min_mean_top10_lift": MIN_PRODUCTION_TOP10_LIFT,
        },
        "feature_columns": list(design.columns),
        "sector_columns": sector_columns,
        "label_column": LABEL_COLUMN,
        "label_horizon_sessions": HORIZON_5,
        "purge_sessions": PURGE_SESSIONS,
        "walk_forward_folds": [r.__dict__ for r in results],
        "dataset_start": str(df["trade_date"].min().date()),
        "dataset_end": str(df["trade_date"].max().date()),
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, path)

    print(f"model_version={MODEL_VERSION}")
    for name, model_results in model_scores.items():
        print(f"summary={name} mean_auc={np.mean([r.auc for r in model_results]):.4f} "
              f"mean_ap={np.mean([r.average_precision for r in model_results]):.4f} "
              f"mean_brier={np.mean([r.brier for r in model_results]):.4f} "
              f"mean_top10_lift={np.mean([r.top_decile_lift for r in model_results]):.2f}")
    print(f"champion={champion_name}")
    print(f"production_ready={ready}")
    print(f"saved_model={path}")
    return results
