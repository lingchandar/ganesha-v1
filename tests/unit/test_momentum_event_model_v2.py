import pandas as pd

from src.prediction.momentum_event_model_v2 import _build_folds, _design_frame, _production_ready


def test_build_folds_keep_purge_gap():
    dates = pd.date_range("2023-01-01", periods=440, freq="D")
    folds = _build_folds(dates, n_folds=4)
    assert len(folds) == 4
    for _, train_idx, val_idx in folds:
        assert train_idx[-1] < val_idx[0]
        assert val_idx[0] - train_idx[-1] >= 5
        assert len(val_idx) == 60


def test_design_frame_keeps_sector_columns_stable():
    frame = pd.DataFrame({
        "sector": ["BANK", "IT", "BANK"],
        "return_1d": [0.01, 0.02, 0.03],
        **{column: [0.0, 0.0, 0.0] for column in __import__("src.prediction.momentum_event_dataset", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS[1:]},
    })
    design, columns = _design_frame(frame)
    assert "sector_BANK" in columns
    assert "sector_IT" in columns
    assert len(design) == 3


def test_production_gate_rejects_weak_results():
    class Result:
        def __init__(self, model_name, auc, lift):
            self.model_name = model_name
            self.auc = auc
            self.top_decile_lift = lift

    weak = [Result("hist_gradient_boosting", 0.55, 1.10)]
    assert _production_ready(weak) is False
