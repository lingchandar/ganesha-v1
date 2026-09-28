"""
GANESHA V1 — Backtest Report Generator

Produces comprehensive terminal tables, markdown artifacts, and JSON/CSV trade logs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from src.backtest.backtest_engine import BacktestResult
from src.backtest.performance import PerformanceSummary


class BacktestReportGenerator:
    """
    Renders backtest metrics into human-readable terminal output,
    markdown reports, and exportable artifacts.
    """

    @staticmethod
    def generate_ascii_summary(result: BacktestResult) -> str:
        """
        Creates an institutional-grade ASCII summary table for terminal logging.
        """
        s = result.summary
        c = result.config

        lines = [
            "==========================================================================================",
            "                          🕉️  GANESHA V1 — HISTORICAL BACKTEST REPORT                     ",
            "==========================================================================================",
            f" Initial Capital: ₹{s.initial_capital_inr:,.2f}          Final Equity: ₹{s.final_equity_inr:,.2f}",
            f" Net Profit:      ₹{s.net_profit_inr:,.2f} ({s.net_return_pct:+.2f}%)   CAGR:         {s.cagr_pct:+.2f}%",
            "------------------------------------------------------------------------------------------",
            " TRADE EXECUTION METRICS:",
            f"   Total Closed Trades:      {s.total_trades:<6} | Winning Trades:         {s.winning_trades} ({s.win_rate_pct:.1f}%)",
            f"   Losing Trades:            {s.losing_trades:<6} | Scratch Trades:         {s.scratch_trades} ({s.loss_rate_pct:.1f}%)",
            f"   Profit Factor:            {s.profit_factor:<6.2f} | Win/Loss Payoff Ratio:  {s.payoff_ratio:.2f}",
            f"   Average Trade PnL:        ₹{s.avg_trade_pnl_inr:<9.2f} | Total Statutory Friction: ₹{s.total_friction_inr:,.2f}",
            "------------------------------------------------------------------------------------------",
            " RISK & EXPECTANCY (R-MULTIPLES):",
            f"   Average Win:              +{s.avg_win_r:<5.2f}R | Average Loss:           -{s.avg_loss_r:.2f}R",
            f"   Expectancy (per trade):   +{s.expectancy_r:<5.2f}R | Net Expectancy (INR):   ₹{s.expectancy_inr:,.2f}",
            "------------------------------------------------------------------------------------------",
            " DRAWDOWN & RISK-ADJUSTED RETURNS:",
            f"   Max Drawdown:             -{s.max_drawdown_pct:<5.2f}% (₹{s.max_drawdown_inr:,.2f})",
            f"   Max Drawdown Duration:    {s.max_drawdown_duration_days} sessions",
            f"   Calmar Ratio:             {s.calmar_ratio:<6.2f} | Sharpe Ratio:           {s.sharpe_ratio:.2f}",
            f"   Sortino Ratio:            {s.sortino_ratio:<6.2f} | Risk-Free Benchmark:     6.50%",
            "------------------------------------------------------------------------------------------",
            " HOLDING DURATION:",
            f"   Average Hold (All):       {s.avg_holding_days_all:.1f} days",
            f"   Average Hold (Winners):   {s.avg_holding_days_winners:.1f} days",
            f"   Average Hold (Losers):    {s.avg_holding_days_losers:.1f} days",
            "==========================================================================================",
        ]

        # Setup Breakdown
        if s.setup_breakdown:
            lines.append(" PERFORMANCE BY SETUP CLASS:")
            lines.append("   Setup Type                  | Trades | Win Rate | Net PnL (INR) | Avg R")
            lines.append("   ----------------------------+--------+----------+---------------+------")
            for setup, m in s.setup_breakdown.items():
                lines.append(
                    f"   {setup:<27} | {m['trades']:<6} | {m['win_rate_pct']:>7.1f}% | ₹{m['net_pnl_inr']:>12,.2f} | {m['avg_r']:>+4.2f}R"
                )
            lines.append("------------------------------------------------------------------------------------------")

        # Exit Reason Breakdown
        if s.exit_reason_breakdown:
            lines.append(" PERFORMANCE BY EXIT REASON:")
            lines.append("   Exit Reason                 | Trades | Win Rate | Net PnL (INR) | Avg R")
            lines.append("   ----------------------------+--------+----------+---------------+------")
            for reason, m in s.exit_reason_breakdown.items():
                lines.append(
                    f"   {reason:<27} | {m['trades']:<6} | {m['win_rate_pct']:>7.1f}% | ₹{m['net_pnl_inr']:>12,.2f} | {m['avg_r']:>+4.2f}R"
                )
            lines.append("==========================================================================================")

        return "\n".join(lines)

    @staticmethod
    def generate_markdown_report(result: BacktestResult) -> str:
        """
        Creates a GitHub-flavored Markdown report.
        """
        s = result.summary
        c = result.config

        md = [
            "# 🕉️ GANESHA V1 — Historical Backtest Report",
            "",
            "### Executive Summary",
            "",
            "| Metric | Value | Institutional Target |",
            "| :--- | :---: | :---: |",
            f"| **Initial Capital** | ₹{s.initial_capital_inr:,.2f} | ₹500,000.00 |",
            f"| **Final Equity** | ₹{s.final_equity_inr:,.2f} | — |",
            f"| **Net Return** | **{s.net_return_pct:+.2f}%** | Positive |",
            f"| **CAGR** | **{s.cagr_pct:+.2f}%** | > 15.0% |",
            f"| **Win Rate** | **{s.win_rate_pct:.1f}%** | > 50.0% |",
            f"| **Profit Factor** | **{s.profit_factor:.2f}** | > 1.50 |",
            f"| **Expectancy** | **+{s.expectancy_r:.2f}R** | > +0.30R |",
            f"| **Max Drawdown** | **-{s.max_drawdown_pct:.2f}%** | < 15.0% |",
            f"| **Calmar Ratio** | **{s.calmar_ratio:.2f}** | > 1.50 |",
            f"| **Sharpe Ratio** | **{s.sharpe_ratio:.2f}** | > 1.00 |",
            f"| **Total Friction Paid** | **₹{s.total_friction_inr:,.2f}** | Deducted strictly |",
            "",
            "---",
            "",
            "### Trade Performance Breakdown",
            "",
            "| Setup Class | Trades | Win Rate | Net PnL (₹) | Avg R |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]

        for setup, m in s.setup_breakdown.items():
            md.append(f"| **{setup}** | {m['trades']} | {m['win_rate_pct']:.1f}% | ₹{m['net_pnl_inr']:,.2f} | {m['avg_r']:+.2f}R |")

        md.extend([
            "",
            "### Exit Distribution",
            "",
            "| Exit Trigger | Trades | Win Rate | Net PnL (₹) | Avg R |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])

        for reason, m in s.exit_reason_breakdown.items():
            md.append(f"| **{reason}** | {m['trades']} | {m['win_rate_pct']:.1f}% | ₹{m['net_pnl_inr']:,.2f} | {m['avg_r']:+.2f}R |")

        return "\n".join(md)

    @staticmethod
    def export_artifacts(
        result: BacktestResult,
        output_dir: Path = Path("audit_logs/backtests"),
    ) -> Path:
        """
        Saves summary JSON, trade log CSV, and equity curve CSV to disk.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp_str}"
        run_dir.mkdir(exist_ok=True)

        # 1. Summary JSON
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(result.summary.to_dict(), f, indent=2)

        # 2. Trades CSV
        if result.trades:
            trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
            trades_df.to_csv(run_dir / "trades.csv", index=False)

        # 3. Equity Curve CSV
        if result.equity_curve:
            eq_df = pd.DataFrame([pt.to_dict() for pt in result.equity_curve])
            eq_df.to_csv(run_dir / "equity_curve.csv", index=False)

        # 4. Markdown Report
        md_content = BacktestReportGenerator.generate_markdown_report(result)
        with open(run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        return run_dir
