"""
GANESHA V1 — Historical Backtesting Engine (Stage 9)
Provides bar-by-bar simulation with zero look-ahead bias, execution realism,
statutory friction deduction, portfolio risk management, and performance analysis.
"""

from src.backtest.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestPosition,
    BacktestResult,
)
from src.backtest.performance import BacktestTrade
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.performance import BacktestPerformanceAnalyzer
from src.backtest.report_generator import BacktestReportGenerator

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestTrade",
    "BacktestPosition",
    "BacktestResult",
    "HistoricalDataLoader",
    "BacktestPerformanceAnalyzer",
    "BacktestReportGenerator",
]
