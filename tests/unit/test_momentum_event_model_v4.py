import numpy as np
import pandas as pd

from src.prediction.momentum_event_model_v4 import (
    LABEL_COLUMN,
    _add_cross_sectional_features,
    _rank_metrics,
    _validate,
)
from src.prediction.momentum_event_dataset import FEATURE_COLUMNS


def _row(date: str, ticker: str, sector: str, value: float, label: int = 0) -> dict:
    row = {column: value for column in FEATURE_COLUMNS}
    row.update(
        {
            "trade_date": pd.Timestamp(date),
            "ticker_symbol": ticker,
            "sector": sector,
            LABEL_COLUMN: label,
        }
    )
    return row


def test_cross_sectional_ranks_are_point_in_time():
    rows = [
        _row("2026-01-02", "A", "BANK", 1.0),
        _row("2026-01-02", "B", "BANK", 2.0),
        _row("2026-01-02", "C", "IT", 3.0),
        _row("2026-01-05", "A", "BANK", 10.0),
        _row("2026-01-05", "B", "BANK", 20.0),
        _row("2026-01-05", "C", "IT", 30.0),
    ]
    df = pd.DataFrame(rows)
    ranked = _add_cross_sectional_features(df)
    first = ranked[ranked.trade_date == pd.Timestamp("2026-01-02")]
    second = ranked[ranked.trade_date == pd.Timestamp("2026-01-05")]
    assert np.allclose(first["rank_return_1d"].to_numpy(), [1 / 3, 2 / 3, 1.0])
    assert np.allclose(second["rank_return_1d"].to_numpy(), [1 / 3, 2 / 3, 1.0])


def test_rank_metrics_lift_uses_validation_base_rate():
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    probability = np.array([0.99, 0.98, 0.97, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14])
    metrics = _rank_metrics(y, probability)
    assert metrics["top10_rate"] == 0.3
    assert metrics["top10_lift"] == 1.0


def test_required_columns_are_validated():
    df = pd.DataFrame([_row("2026-01-02", "A", "BANK", 1.0)])
    _validate(df)
    broken = df.drop(columns=[FEATURE_COLUMNS[0]])
    try:
        _validate(broken)
    except ValueError as exc:
        assert FEATURE_COLUMNS[0] in str(exc)
    else:
        raise AssertionError("Expected missing-column validation error")
