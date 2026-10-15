"""Leakage-safe walk-forward model for early momentum-event prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.momentum_event_dataset import FEATURE_COLUMNS, HORIZON_5

LABEL_COLUMN = "hit_up_before_down_5d"
MODEL_VERSION = "momentum_event_v1"
PURGE_SESSIONS = HORIZON_5
N_FOLDS = 4
MIN_TRAIN_SESSIONS = 180
VALIDATION_SESSIONS = 60


@dataclass(frozen=True)
class FoldResult:
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

    # Use the final n_folds validation windows, each preceded by an expanding
    # training window. The validation windows are chronological and disjoint.
    available = len(unique_dates)
    total_validation = n_folds * VALIDATION_SESSIONS
    first_validation = available - total_validation
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        val_start = first_validation + fold * VALIDATION_SESSIONS
        val_end = val_start + VALIDATION_SESSIONS
        train_end = val_start - PURGE_SESSIONS
        if train_end < MIN_TRAIN_SESSIONS:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((fold + 1, train_idx, val_idx))
    if len(folds) < 2:
        raise ValueError("Walk-forward configuration produced fewer than two folds")
    return folds


def _make_model() -> CalibratedClassifierCV:
    base = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=3.0,
        random_state=42,
    )
    return CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )


def _validate_frame(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [LABEL_COLUMN, "ticker_symbol", "trade_date"]
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


def train_walk_forward(dataset_path: str, model_path: str) -> list[FoldResult]:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    df = df.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    _validate_frame(df)

    dates = _normal_dates(df)
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    date_to_position = {value: pos for pos, value in enumerate(unique_dates)}
    date_positions = dates.map(date_to_position).to_numpy()

    results: list[FoldResult] = []
    oof_probability = np.full(len(df), np.nan, dtype=float)

    for fold, train_date_idx, val_date_idx in _build_folds(unique_dates):
        val_start_pos = int(val_date_idx[0])
        val_end_pos = int(val_date_idx[-1])
        train_end_pos = int(train_date_idx[-1])

        train_mask = date_positions <= train_end_pos
        val_mask = (date_positions >= val_start_pos) & (date_positions <= val_end_pos)
        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()

        # Explicit invariant: no training label interval may reach the first
        # validation decision date. Labels resolve up to 5 sessions later.
        if train_df["trade_date"].max() >= val_df["trade_date"].min():
            raise AssertionError("Training data reaches validation period")
        if train_df.empty or val_df.empty or train_df[LABEL_COLUMN].nunique() < 2:
            raise ValueError(f"Fold {fold} has insufficient training data/classes")
        if val_df[LABEL_COLUMN].nunique() < 2:
            raise ValueError(f"Fold {fold} validation contains one class only")

        model = _make_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
        probability = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
        oof_probability[val_mask] = probability

        auc = roc_auc_score(val_df[LABEL_COLUMN], probability)
        ap = average_precision_score(val_df[LABEL_COLUMN], probability)
        brier = brier_score_loss(val_df[LABEL_COLUMN], probability)
        top_rate, lift = _top_decile_metrics(val_df[LABEL_COLUMN].reset_index(drop=True), probability)
        results.append(
            FoldResult(
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
        )

    # Final production candidate is trained on all historical observations,
    # while the fold metrics above remain the honest model-selection evidence.
    final_model = _make_model()
    final_model.fit(df[FEATURE_COLUMNS], df[LABEL_COLUMN])
    output = {
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "label_horizon_sessions": HORIZON_5,
        "purge_sessions": PURGE_SESSIONS,
        "model": final_model,
        "walk_forward_folds": [result.__dict__ for result in results],
        "dataset_start": str(df["trade_date"].min().date()),
        "dataset_end": str(df["trade_date"].max().date()),
        "oof_predictions": pd.DataFrame({
            "ticker_symbol": df["ticker_symbol"],
            "trade_date": df["trade_date"],
            "label": df[LABEL_COLUMN],
            "oof_probability": oof_probability,
        }),
    }
    path = joblib.Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, path)

    for result in results:
        print(
            f"fold={result.fold} train={result.train_start}:{result.train_end} "
            f"validation={result.validation_start}:{result.validation_end} "
            f"train_rows={result.train_rows} validation_rows={result.validation_rows} "
            f"positive_rate={result.validation_positive_rate:.4f} "
            f"auc={result.auc:.4f} ap={result.average_precision:.4f} "
            f"brier={result.brier:.4f} top10_rate={result.top_decile_rate:.4f} "
            f"top10_lift={result.top_decile_lift:.2f}"
        )
    print(f"model_version={MODEL_VERSION}")
    print(f"folds={len(results)}")
    print(f"mean_auc={np.mean([r.auc for r in results]):.4f}")
    print(f"mean_average_precision={np.mean([r.average_precision for r in results]):.4f}")
    print(f"mean_brier={np.mean([r.brier for r in results]):.4f}")
    print(f"mean_top10_lift={np.mean([r.top_decile_lift for r in results]):.2f}")
    print(f"saved_model={path}")
    return results
