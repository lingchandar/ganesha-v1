from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_entry_prediction_timeouts_v1 import audit


def test_timeout_audit_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"ticker_symbol": ["TCS"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Dataset missing columns"):
        audit(path)


def test_timeout_audit_handles_no_unresolved(tmp_path: Path, capsys):
    path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2026-01-02",
                "ticker_symbol": "TCS",
                "setup_type": "DEMAND_SUPPORT_BOUNCE",
                "entry_price": 100.0,
                "risk_points": 5.0,
                "outcome_valid": 1,
                "label_target_before_stop": 1,
                "mfe_r": 2.1,
                "mae_r": -0.2,
            }
        ]
    ).to_csv(path, index=False)

    audit(path)
    assert "unresolved_rows=0" in capsys.readouterr().out
