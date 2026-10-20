import pandas as pd

from src.prediction.momentum_event_model_v3 import (
    _build_folds,
    _design_frame,
    _production_ready,
)


def test_build_folds_have_exact_purge_gap_and_validation_window():
    dates = pd.date_range("2023-01-01", periods=440, freq="D")
    folds = _build_folds(dates, n_folds=4)
    assert len(folds) == 4
    for _, train_idx, val_idx in folds:
        assert len(val_idx) == 60
        assert val_idx[0] - train_idx[-1] >= 3


def test_design_frame_keeps_sector_columns_stable():
    from src.prediction.momentum_event_dataset import FEATURE_COLUMNS

    frame = pd.DataFrame({
        "sector": ["BANK", "IT", "BANK"],
        **{column: [0.0, 0.0, 0.0] for column in FEATURE_COLUMNS},
    })
    design, columns = _design_frame(frame)
    assert "sector_BANK" in columns
    assert "sector_IT" in columns
    assert len(design) == 3


def test_production_gate_requires_auc_and_ranked_lift():
    class Result:
        def __init__(self, model_name, auc, top5_lift, top10_lift):
            self.model_name = model_name
            self.auc = auc
            self.top5_lift = top5_lift
            self.top10_lift = top10_lift

    weak_auc = [Result("x", 0.59, 2.0, 1.6)]
    weak_lift = [Result("x", 0.65, 1.5, 1.6)]
    strong = [Result("x", 0.62, 1.9, 1.5)]
    assert _production_ready(weak_auc) is False
    assert _production_ready(weak_lift) is False
    assert _production_ready(strong) is True
