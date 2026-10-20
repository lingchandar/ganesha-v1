"""Momentum Event Model v4 with cross-sectional, point-in-time ranking.

v3 showed only modest out-of-sample lift. v4 keeps the same event label and
walk-forward discipline but adds daily cross-sectional ranks and a small set
of point-in-time interaction features. This makes the model focus on what is
unusually strong *today* across the 59-stock universe instead of relying on
absolute feature levels that can drift across market regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

from src.prediction.momentum_event_dataset import FEATURE_COLUMNS
from src.prediction.momentum_event_dataset_v2 import EVENT_HORIZON

LABEL_COLUMN = "early_momentum_event_3d"
MODEL_VERSION = "momentum_event_v4"
PURGE_SESSIONS = EVENT_HORIZON
CALIBRATION_GAP = EVENT_HORIZON
N_FOLDS = 4
VALIDATION_SESSIONS = 60
MIN_TRAIN_SESSIONS = 180
TOP_FRACTIONS = (0.01, 0.03, 0.05, 0.10)

# Keep the gate conservative: v4 is not production-ready unless the ranked
# event capture is consistently strong across chronological folds.
MIN_MEAN_AUC = 0.60
MIN_MEAN_TOP5_LIFT = 1.75
MIN_MEAN_TOP10_LIFT = 1.40

RANK_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "range_atr_multiple",
    "atr_expansion_ratio",
    "relative_strength_20d",
    "sector_strength_20d",
    "delivery_pct_ratio_20d",
    "distance_to_20d_high_pct",
    "distance_to_50d_high_pct",
]


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


def _build_folds(unique_dates: pd.DatetimeIndex) -> list[tuple[int, int, int]]:
    required = MIN_TRAIN_SESSIONS + N_FOLDS * VALIDATION_SESSIONS
    if len(unique_dates) < required:
        raise ValueError(f"Not enough sessions: need {required}, got {len(unique_dates)}")
    first_validation = len(unique_dates) - N_FOLDS * VALIDATION_SESSIONS
    folds: list[tuple[int, int, int]] = []
    for fold in range(N_FOLDS):
        val_start = first_validation + fold * VALIDATION_SESSIONS
        val_end = val_start + VALIDATION_SESSIONS
        train_end = val_start - PURGE_SESSIONS
        if train_end >= MIN_TRAIN_SESSIONS:
            folds.append((fold + 1, train_end, val_start))
    if len(folds) < 2:
        raise ValueError("Fewer than two walk-forward folds were produced")
    return folds


def _validate(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [LABEL_COLUMN, "ticker_symbol", "trade_date", "sector"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Momentum event v4 dataset contains NaN in required columns")
    if df.duplicated(["ticker_symbol", "trade_date"]).any():
        raise ValueError("Momentum event v4 dataset contains duplicate ticker/date rows")
    if not set(df[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} must be binary")


def _add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_group = out.groupby("trade_date", sort=False)
    for column in RANK_FEATURES:
        out[f"rank_{column}"] = date_group[column].rank(method="average", pct=True)

    # Point-in-time interactions: all inputs are known at the event close.
    out["volume_price_impulse"] = out["volume_zscore_20d"] * out["return_1d"]
    out["volume_range_impulse"] = out["volume_ratio_20d"] * out["range_atr_multiple"]
    out["volatility_momentum_impulse"] = out["atr_expansion_ratio"] * out["return_3d"]
    out["relative_sector_alignment"] = out["relative_strength_20d"] * out["sector_strength_20d"]
    out["breakout_proximity"] = (
        out["distance_to_20d_high_pct"] + out["distance_to_50d_high_pct"]
    ) / 2.0
    out["rank_momentum_volume"] = (
        out["rank_return_5d"] + out["rank_volume_ratio_20d"] + out["rank_relative_strength_20d"]
    ) / 3.0
    out["rank_momentum_quality"] = (
        out["rank_return_3d"] + out["rank_atr_expansion_ratio"]
        + out["rank_volume_zscore_20d"] + out["rank_sector_strength_20d"]
    ) / 4.0
    return out


def _design_frame(df: pd.DataFrame, sector_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    feature_df = _add_cross_sectional_features(df)
    extra = [c for c in feature_df.columns if c.startswith("rank_") or c.endswith("_impulse") or c in {"breakout_proximity"}]
    numeric_columns = list(dict.fromkeys(FEATURE_COLUMNS + extra))
    numeric = feature_df[numeric_columns].astype(float)
    sector = pd.get_dummies(feature_df["sector"].fillna("UNKNOWN").astype(str), prefix="sector", dtype=float)
    if sector_columns is None:
        sector_columns = sorted(sector.columns.tolist())
    sector = sector.reindex(columns=sector_columns, fill_value=0.0)
    return pd.concat([numeric, sector], axis=1), sector_columns


def _models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.08, class_weight="balanced", max_iter=1500, random_state=42)),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=8.0,
            random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            max_depth=9,
            min_samples_leaf=45,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }


def _calibrated(model: object) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        model,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3, gap=CALIBRATION_GAP),
    )


def _rank_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    y = y.reset_index(drop=True)
    base = float(y.mean())
    result: dict[str, float] = {}
    for fraction in TOP_FRACTIONS:
        key = int(fraction * 100)
        count = max(1, int(np.ceil(len(y) * fraction)))
        order = np.argsort(-probability)[:count]
        rate = float(y.iloc[order].mean())
        result[f"top{key}_rate"] = rate
        result[f"top{key}_lift"] = rate / base if base else float("nan")
    return result


def _ready(results: list[FoldResult]) -> bool:
    grouped: dict[str, list[FoldResult]] = {}
    for row in results:
        grouped.setdefault(row.model_name, []).append(row)
    return any(
        np.mean([r.auc for r in rows]) >= MIN_MEAN_AUC
        and np.mean([r.top5_lift for r in rows]) >= MIN_MEAN_TOP5_LIFT
        and np.mean([r.top10_lift for r in rows]) >= MIN_MEAN_TOP10_LIFT
        for rows in grouped.values()
    )


def train_walk_forward(dataset_path: str, model_path: str) -> list[FoldResult]:
    df = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    df = df.sort_values(["trade_date", "ticker_symbol"]).reset_index(drop=True)
    _validate(df)
    design, sector_columns = _design_frame(df)
    unique_dates = pd.DatetimeIndex(sorted(df["trade_date"].dt.normalize().unique()))
    date_position = {date: i for i, date in enumerate(unique_dates)}
    positions = df["trade_date"].dt.normalize().map(date_position).to_numpy()

    results: list[FoldResult] = []
    scores: dict[str, list[FoldResult]] = {}
    for model_name in _models():
        for fold, train_end, val_start in _build_folds(unique_dates):
            val_end = val_start + VALIDATION_SESSIONS
            train_mask = positions < train_end
            val_mask = (positions >= val_start) & (positions < val_end)
            train_df = df.loc[train_mask]
            val_df = df.loc[val_mask]
            if train_df["trade_date"].max() >= val_df["trade_date"].min():
                raise AssertionError("Training data reaches validation period")
            if train_df[LABEL_COLUMN].nunique() < 2 or val_df[LABEL_COLUMN].nunique() < 2:
                raise ValueError(f"Fold {fold} has insufficient class diversity")

            model = _calibrated(_models()[model_name])
            model.fit(design.loc[train_mask], train_df[LABEL_COLUMN])
            probability = model.predict_proba(design.loc[val_mask])[:, 1]
            ranked = _rank_metrics(val_df[LABEL_COLUMN], probability)
            row = FoldResult(
                model_name=model_name, fold=fold,
                train_start=str(train_df.trade_date.min().date()),
                train_end=str(train_df.trade_date.max().date()),
                validation_start=str(val_df.trade_date.min().date()),
                validation_end=str(val_df.trade_date.max().date()),
                train_rows=len(train_df), validation_rows=len(val_df),
                purge_rows=int(((positions >= train_end) & (positions < val_start)).sum()),
                validation_positive_rate=float(val_df[LABEL_COLUMN].mean()),
                auc=float(roc_auc_score(val_df[LABEL_COLUMN], probability)),
                average_precision=float(average_precision_score(val_df[LABEL_COLUMN], probability)),
                brier=float(brier_score_loss(val_df[LABEL_COLUMN], probability)),
                top1_rate=ranked["top1_rate"], top1_lift=ranked["top1_lift"],
                top3_rate=ranked["top3_rate"], top3_lift=ranked["top3_lift"],
                top5_rate=ranked["top5_rate"], top5_lift=ranked["top5_lift"],
                top10_rate=ranked["top10_rate"], top10_lift=ranked["top10_lift"],
            )
            results.append(row)
            scores.setdefault(model_name, []).append(row)
            print(
                f"model={model_name} fold={fold} auc={row.auc:.4f} ap={row.average_precision:.4f} "
                f"brier={row.brier:.4f} top1={row.top1_rate:.4f}/{row.top1_lift:.2f} "
                f"top3={row.top3_rate:.4f}/{row.top3_lift:.2f} "
                f"top5={row.top5_rate:.4f}/{row.top5_lift:.2f} "
                f"top10={row.top10_rate:.4f}/{row.top10_lift:.2f}"
            )

    ranked_models = sorted(
        scores.items(),
        key=lambda item: (np.mean([r.top5_lift for r in item[1]]), np.mean([r.auc for r in item[1]])),
        reverse=True,
    )
    champion_name = ranked_models[0][0]
    champion = _calibrated(_models()[champion_name])
    champion.fit(design, df[LABEL_COLUMN])
    production_ready = _ready(results)

    artifact = {
        "model_version": MODEL_VERSION,
        "model": champion,
        "model_name": champion_name,
        "production_ready": production_ready,
        "feature_columns": list(design.columns),
        "sector_columns": sector_columns,
        "label_column": LABEL_COLUMN,
        "event_horizon_sessions": EVENT_HORIZON,
        "purge_sessions": PURGE_SESSIONS,
        "calibration_gap": CALIBRATION_GAP,
        "rank_features": RANK_FEATURES,
        "top_fractions": TOP_FRACTIONS,
        "production_gate": {
            "min_mean_auc": MIN_MEAN_AUC,
            "min_mean_top5_lift": MIN_MEAN_TOP5_LIFT,
            "min_mean_top10_lift": MIN_MEAN_TOP10_LIFT,
        },
        "walk_forward_folds": [r.__dict__ for r in results],
        "dataset_start": str(df.trade_date.min().date()),
        "dataset_end": str(df.trade_date.max().date()),
        "dataset_rows": len(df),
        "event_rate": float(df[LABEL_COLUMN].mean()),
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    print(f"model_version={MODEL_VERSION}")
    for name, rows in scores.items():
        print(
            f"summary={name} mean_auc={np.mean([r.auc for r in rows]):.4f} "
            f"mean_ap={np.mean([r.average_precision for r in rows]):.4f} "
            f"mean_brier={np.mean([r.brier for r in rows]):.4f} "
            f"mean_top5_lift={np.mean([r.top5_lift for r in rows]):.2f} "
            f"mean_top10_lift={np.mean([r.top10_lift for r in rows]):.2f}"
        )
    print(f"champion={champion_name}")
    print(f"production_ready={production_ready}")
    print(f"saved_model={path}")
    return results
