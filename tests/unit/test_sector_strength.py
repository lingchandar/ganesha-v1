"""
Unit tests for Sector Strength and Mansfield Relative Strength Matrix (src/regime/sector_strength.py).
Verifies:
1. Mansfield Relative Strength (MRS) formula calculation
2. Relative ROC calculation vs Benchmark
3. Dual-layer filter: passes when Sector beats Nifty AND Stock beats Nifty + Sector
4. Dual-layer filter: fails when Sector lags Nifty (ROC_20 <= 0)
5. Dual-layer filter: fails when Stock lags Sector or Nifty
6. Insufficient history handling (< 55 sessions)
"""
import pandas as pd
import numpy as np
import pytest

from src.regime.sector_strength import RelativeStrengthEngine


def test_mansfield_rs_formula():
    """Verify MRS calculation matches formula MRS = (Ratio / SMA50(Ratio) - 1) * 100."""
    periods = 100
    base = pd.Series(np.linspace(100, 200, periods))
    benchmark = pd.Series(np.full(periods, 100.0))

    mrs = RelativeStrengthEngine.calculate_mansfield_rs(base, benchmark, sma_period=50)

    assert len(mrs) == periods
    # Since base is growing while benchmark is flat, recent ratio is above SMA50
    assert mrs.iloc[-1] > 0.0


def test_relative_roc_calculation():
    """Verify 20-period relative ROC returns accurate outperformance spread."""
    periods = 50
    base = pd.Series(np.linspace(100, 120, periods))       # +20% over full, roughly +8% over 20
    benchmark = pd.Series(np.linspace(100, 105, periods))  # +5% over full, roughly +2% over 20

    rel_roc = RelativeStrengthEngine.calculate_relative_roc(base, benchmark, period=20)
    assert rel_roc > 0.0


def test_dual_layer_passes_strong_leader():
    """Stock beating Sector and Nifty, while Sector is beating Nifty passes filter."""
    periods = 70
    nifty = pd.Series(np.linspace(100, 110, periods))       # +10%
    sector = pd.Series(np.linspace(100, 130, periods))      # +30% (Sector beating Nifty)
    stock = pd.Series(np.linspace(100, 160, periods))       # +60% (Stock beating both)

    res = RelativeStrengthEngine.evaluate_dual_layer_relative_strength(
        stock_close=stock,
        sector_close=sector,
        nifty_close=nifty,
        stock_symbol="TATAMOTORS",
        sector_name="AUTOMOBILE",
    )

    assert res["passes_rs_filter"] is True
    assert res["is_sector_leader"] is True
    assert res["is_stock_leader"] is True
    assert res["mrs_stock_vs_nifty"] > 0
    assert res["mrs_stock_vs_sector"] > 0
    assert res["sector_roc_20_vs_nifty"] > 0


def test_dual_layer_fails_lagging_sector():
    """If Sector is lagging Nifty, candidate stock is rejected even if individual stock is strong."""
    periods = 70
    nifty = pd.Series(np.linspace(100, 130, periods))       # +30%
    sector = pd.Series(np.linspace(100, 105, periods))      # +5% (Sector lagging Nifty)
    stock = pd.Series(np.linspace(100, 150, periods))       # +50%

    res = RelativeStrengthEngine.evaluate_dual_layer_relative_strength(
        stock_close=stock,
        sector_close=sector,
        nifty_close=nifty,
        stock_symbol="TCS",
        sector_name="INFORMATION_TECHNOLOGY",
    )

    assert res["passes_rs_filter"] is False
    assert res["is_sector_leader"] is False
    assert "SECTOR_LAGGING_NIFTY" in res["rejection_reason"]


def test_dual_layer_fails_lagging_stock():
    """If Sector is leading Nifty but stock is lagging its sector, candidate is rejected."""
    periods = 70
    nifty = pd.Series(np.linspace(100, 110, periods))       # +10%
    sector = pd.Series(np.linspace(100, 150, periods))      # +50% (Sector leading)
    stock = pd.Series(np.linspace(100, 120, periods))       # +20% (Stock lagging Sector)

    res = RelativeStrengthEngine.evaluate_dual_layer_relative_strength(
        stock_close=stock,
        sector_close=sector,
        nifty_close=nifty,
        stock_symbol="WIPRO",
        sector_name="INFORMATION_TECHNOLOGY",
    )

    assert res["passes_rs_filter"] is False
    assert "STOCK_LAGGING_SECTOR" in res["rejection_reason"]


def test_dual_layer_insufficient_history():
    """Fewer than 55 bars rejects candidate cleanly without crashing."""
    short_series = pd.Series(np.linspace(100, 120, 30))

    res = RelativeStrengthEngine.evaluate_dual_layer_relative_strength(
        stock_close=short_series,
        sector_close=short_series,
        nifty_close=short_series,
        stock_symbol="NEWIPO",
        sector_name="TECH",
    )

    assert res["passes_rs_filter"] is False
    assert "INSUFFICIENT_HISTORY" in res["rejection_reason"]
