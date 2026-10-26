import pandas as pd
import pytest

from scripts.analyze_entry_prediction_timeout_distribution_v1 import _summary, _validate


def test_summary_reports_timeout_distribution():
    df = pd.DataFrame({"horizon_end_r": [-0.6, -0.1, 0.1, 0.3, 0.6, 1.2]})
    result = _summary(df)
    assert result["rows"] == 6
    assert result["mean_r"] == pytest.approx(0.25)
    assert result["median_r"] == pytest.approx(0.2)
    assert result["positive_rate"] == pytest.approx(4 / 6)
    assert result["ge_0_25r"] == pytest.approx(3 / 6)
    assert result["ge_0_50r"] == pytest.approx(2 / 6)
    assert result["ge_1r"] == pytest.approx(1 / 6)
    assert result["le_neg_0_25r"] == pytest.approx(1 / 6)
    assert result["le_neg_0_50r"] == pytest.approx(1 / 6)


def test_validate_requires_horizon_end_r():
    with pytest.raises(ValueError, match="horizon_end_r"):
        _validate(pd.DataFrame({"trade_date": []}))
