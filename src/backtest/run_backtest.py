"""
GANESHA V1 — Backtest Runner CLI

Usage:
    .venv/bin/python -m src.backtest.run_backtest
    .venv/bin/python -m src.backtest.run_backtest --capital 1000000 --max-positions 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from loguru import logger

from src.backtest.backtest_engine import BacktestEngine, BacktestConfig
from src.backtest.report_generator import BacktestReportGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GANESHA V1 Institutional Historical Backtest.")
    parser.add_argument("--capital", type=float, default=500_000.0, help="Initial capital in INR (default: 500,000)")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="Max risk fraction per trade (default: 0.01 = 1%)")
    parser.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions (default: 5)")
    parser.add_argument("--max-sector", type=int, default=2, help="Max positions per sector (default: 2)")
    parser.add_argument("--no-friction", action="store_true", help="Disable statutory delivery friction deduction")
    parser.add_argument("--no-regime", action="store_true", help="Disable Nifty 50 regime filter")
    parser.add_argument("--no-rs", action="store_true", help="Disable Mansfield RS filter")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--export", action="store_true", default=True, help="Export artifacts to audit_logs/backtests")

    args = parser.parse_args()

    s_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    e_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    config = BacktestConfig(
        initial_capital_inr=args.capital,
        max_risk_pct=args.risk_pct,
        max_concurrent_positions=args.max_positions,
        max_per_sector=args.max_sector,
        enable_friction=not args.no_friction,
        enable_regime_filter=not args.no_regime,
        enable_rs_filter=not args.no_rs,
        start_date=s_date,
        end_date=e_date,
    )

    logger.info("Initializing GANESHA V1 Backtest Engine...")
    engine = BacktestEngine(config=config)
    result = engine.run()

    # Render ASCII Summary
    print("\n" + BacktestReportGenerator.generate_ascii_summary(result) + "\n")

    if args.export:
        run_dir = BacktestReportGenerator.export_artifacts(result)
        logger.success(f"Backtest artifacts exported to: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
