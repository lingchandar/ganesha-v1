"""
GANESHA V1 — Sector Strength & Mansfield Relative Strength Matrix (Module 5)

REAL-TIME TRADING PHILOSOPHY:
Capital in Indian equity markets rotates systematically among leading sectoral indices:
(Nifty Bank, Nifty IT, Nifty Auto, Nifty Pharma, Nifty Metal, Nifty FMCG, Nifty Energy, Nifty Realty).

Hierarchy of Relative Strength:
Broad Market (Nifty 50) -> Sector Index -> Industry -> Individual Stock

Mansfield Relative Strength (MRS) Formulation:
    Ratio(t) = Price_Base(t) / Price_Benchmark(t)
    MRS(t)   = [ Ratio(t) / SMA_50(Ratio(t)) - 1 ] * 100

Dual-Layer Relative Strength Filter:
A candidate stock ONLY qualifies if:
1. Sector Outperformance: Sector index ROC_20 > 0 relative to Nifty 50.
2. Dual MRS Outperformance: Stock demonstrates MRS > 0 against BOTH Nifty 50 AND its sector index.
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger


class RelativeStrengthEngine:
    """
    Computes Mansfield Relative Strength and evaluates the dual-layer sector filter.
    """

    @staticmethod
    def calculate_mansfield_rs(
        base_series: pd.Series,
        benchmark_series: pd.Series,
        sma_period: int = 50
    ) -> pd.Series:
        """
        Calculate Mansfield Relative Strength series.
        MRS > 0 indicates base is outperforming benchmark over the trailing sma_period.
        """
        ratio = base_series / benchmark_series.replace(0, np.nan)
        ratio_sma = ratio.rolling(window=sma_period).mean()
        mrs = ((ratio / ratio_sma) - 1.0) * 100.0
        return mrs.fillna(0.0)

    @staticmethod
    def calculate_relative_roc(
        base_series: pd.Series,
        benchmark_series: pd.Series,
        period: int = 20
    ) -> float:
        """
        Calculate 20-period Relative Rate of Change (ROC) of Base vs Benchmark.
        ROC_20 = (Base_return_20 - Benchmark_return_20)
        """
        base_return = (base_series.iloc[-1] / base_series.iloc[-period] - 1.0) * 100.0
        benchmark_return = (benchmark_series.iloc[-1] / benchmark_series.iloc[-period] - 1.0) * 100.0
        return float(base_return - benchmark_return)

    @classmethod
    def evaluate_dual_layer_relative_strength(
        cls,
        stock_close: pd.Series,
        sector_close: pd.Series,
        nifty_close: pd.Series,
        stock_symbol: str = "STOCK",
        sector_name: str = "SECTOR",
    ) -> Dict[str, any]:
        """
        Evaluate full dual-layer RS gate:
        Layer 1: Sector vs Nifty 50 (ROC_20 > 0)
        Layer 2: Stock vs Nifty 50 (MRS_50 > 0) AND Stock vs Sector (MRS_50 > 0)

        Returns:
            Dict containing boolean pass/fail status, individual MRS numbers, and diagnostic logs.
        """
        min_len = min(len(stock_close), len(sector_close), len(nifty_close))
        if min_len < 55:
            return {
                "passes_rs_filter": False,
                "rejection_reason": f"INSUFFICIENT_HISTORY_FOR_RS: Minimum 55 sessions needed, got {min_len}.",
                "mrs_stock_vs_nifty": 0.0,
                "mrs_stock_vs_sector": 0.0,
                "sector_roc_20_vs_nifty": 0.0,
            }

        # 1. Sector vs Nifty ROC 20
        sector_roc_vs_nifty = cls.calculate_relative_roc(sector_close, nifty_close, period=20)
        sector_outperforming = sector_roc_vs_nifty > 0.0

        # 2. Stock vs Nifty MRS
        stock_vs_nifty_mrs = cls.calculate_mansfield_rs(stock_close, nifty_close, sma_period=50).iloc[-1]

        # 3. Stock vs Sector MRS
        stock_vs_sector_mrs = cls.calculate_mansfield_rs(stock_close, sector_close, sma_period=50).iloc[-1]

        # Dual condition
        stock_outperforming = (stock_vs_nifty_mrs > 0.0) and (stock_vs_sector_mrs > 0.0)
        passes_dual_layer = sector_outperforming and stock_outperforming

        rejection_reason = None
        if not sector_outperforming:
            rejection_reason = (
                f"SECTOR_LAGGING_NIFTY: {sector_name} 20-day relative ROC is {sector_roc_vs_nifty:.2f}% (<= 0)."
            )
        elif not (stock_vs_nifty_mrs > 0.0):
            rejection_reason = (
                f"STOCK_LAGGING_NIFTY: {stock_symbol} Mansfield RS vs Nifty is {stock_vs_nifty_mrs:.2f} (<= 0)."
            )
        elif not (stock_vs_sector_mrs > 0.0):
            rejection_reason = (
                f"STOCK_LAGGING_SECTOR: {stock_symbol} Mansfield RS vs {sector_name} is {stock_vs_sector_mrs:.2f} (<= 0)."
            )

        return {
            "passes_rs_filter": bool(passes_dual_layer),
            "rejection_reason": rejection_reason,
            "mrs_stock_vs_nifty": round(float(stock_vs_nifty_mrs), 2),
            "mrs_stock_vs_sector": round(float(stock_vs_sector_mrs), 2),
            "sector_roc_20_vs_nifty": round(float(sector_roc_vs_nifty), 2),
            "is_sector_leader": bool(sector_outperforming),
            "is_stock_leader": bool(stock_outperforming),
        }
