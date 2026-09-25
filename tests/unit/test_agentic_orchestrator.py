"""
Unit tests for the Master Agentic AI Orchestrator and Forensic Audit Logger.
Verifies the multi-step orchestration pipeline, regime halt, zero-compromise rule,
and audit trail generation.
"""
from datetime import date
import pandas as pd
import numpy as np
import pytest
from src.agentic.orchestrator import GaneshaAgenticOrchestrator
from src.audit.audit_logger import AuditTrailLogger


def _generate_synthetic_nifty_candles(is_bullish: bool = True, n_candles: int = 220) -> pd.DataFrame:
    """Generate synthetic Nifty 50 candle series for testing."""
    dates = pd.date_range(end="2026-09-25", periods=n_candles)
    if is_bullish:
        # Upward trending series: 18000 -> 25000
        close = np.linspace(18000, 25000, n_candles)
    else:
        # Downward trending series: 25000 -> 18000 (Bearish)
        close = np.linspace(25000, 18000, n_candles)

    high = close + 50.0
    low = close - 50.0
    open_p = close - 10.0
    volume = np.full(n_candles, 5000000)

    return pd.DataFrame({
        "timestamp": dates,
        "open_price": open_p,
        "high_price": high,
        "low_price": low,
        "close_price": close,
        "volume_traded": volume,
    })


def test_orchestrator_regime_halt(tmp_path):
    """If Nifty is Bearish, orchestrator must halt all swing generation immediately."""
    bearish_nifty = _generate_synthetic_nifty_candles(is_bullish=False)
    orchestrator = GaneshaAgenticOrchestrator()
    orchestrator.audit_logger = AuditTrailLogger(log_dir=str(tmp_path))

    res = orchestrator.run_swing_scan(
        nifty_history_df=bearish_nifty,
        candidate_data={},
        trade_date=date(2026, 9, 25),
    )

    assert res["status"] == "REGIME_HALT"
    assert res["regime"] == "BEARISH"
    assert res["approved_watchlist"] == []
    assert len(res["scan_id"]) > 0

    # Verify audit file was written
    audit_files = list(tmp_path.glob("scan_*.json"))
    assert len(audit_files) == 1


def test_orchestrator_zero_compromise_rule(tmp_path):
    """If Nifty is Bullish but no candidates pass filters, outputs NO_VALID_SETUP_DETECTED."""
    bullish_nifty = _generate_synthetic_nifty_candles(is_bullish=True)
    orchestrator = GaneshaAgenticOrchestrator()
    orchestrator.audit_logger = AuditTrailLogger(log_dir=str(tmp_path))

    # Candidate with empty or flat candles that don't form any of the 6 setups
    dates = pd.date_range(end="2026-09-25", periods=60)
    flat_stock = pd.DataFrame({
        "open_price": np.full(60, 100.0),
        "high_price": np.full(60, 100.5),
        "low_price": np.full(60, 99.5),
        "close_price": np.full(60, 100.0),
        "volume_traded": np.full(60, 10000),
    })

    candidate_data = {
        "NSE:FLATCO-EQ": {
            "ohlcv": flat_stock,
            "delivery_history": [1000, 1000, 1000, 1000, 1000],
            "today_delivery": 1000,
        }
    }

    res = orchestrator.run_swing_scan(
        nifty_history_df=bullish_nifty,
        candidate_data=candidate_data,
        trade_date=date(2026, 9, 25),
    )

    assert res["status"] == "NO_VALID_SETUP_DETECTED"
    assert res["approved_watchlist"] == []
    assert res["rejected_count"] >= 1
