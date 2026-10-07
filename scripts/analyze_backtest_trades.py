"""
GANESHA V1 — Backtest Trade Diagnostics

Analyzes an exported trades.csv before strategy changes are made.
This is intentionally diagnostic only: it does not modify strategy parameters.

Usage:
    .venv/bin/python scripts/analyze_backtest_trades.py
    .venv/bin/python scripts/analyze_backtest_trades.py --trades-csv audit_logs/backtests/run_YYYYMMDD_HHMMSS/trades.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "ticker_symbol",
    "setup_type",
    "exit_reason",
    "net_pnl_inr",
    "realized_r_multiple",
    "holding_days",
}


def find_latest_trades_csv(root: Path = Path("audit_logs/backtests")) -> Path:
    """Return the newest exported trades.csv."""
    candidates = sorted(root.glob("run_*/trades.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No trades.csv found under {root}. Run the backtest with artifact export first."
        )
    return candidates[0]


def load_trades(path: Path) -> pd.DataFrame:
    """Load and validate the exported trade log."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Trade log is missing required columns: {sorted(missing)}")

    for column in ("net_pnl_inr", "realized_r_multiple", "holding_days"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["net_pnl_inr", "realized_r_multiple", "holding_days"]].isna().any().any():
        raise ValueError("Trade log contains non-numeric values in required metric columns")

    return df


def print_group_breakdown(df: pd.DataFrame, column: str, title: str) -> None:
    """Print a compact performance breakdown for a categorical field."""
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            trades=("net_pnl_inr", "size"),
            wins=("net_pnl_inr", lambda s: int((s > 0).sum())),
            net_pnl_inr=("net_pnl_inr", "sum"),
            avg_r=("realized_r_multiple", "mean"),
            avg_hold_days=("holding_days", "mean"),
        )
        .sort_values("net_pnl_inr", ascending=False)
    )
    grouped["win_rate_pct"] = grouped["wins"] / grouped["trades"] * 100.0

    print(f"\n{title}")
    print("-" * 96)
    print(
        grouped[
            ["trades", "win_rate_pct", "net_pnl_inr", "avg_r", "avg_hold_days"]
        ].to_string(
            formatters={
                "win_rate_pct": "{:.1f}%".format,
                "net_pnl_inr": "₹{:,.2f}".format,
                "avg_r": "{:+.2f}R".format,
                "avg_hold_days": "{:.1f}".format,
            }
        )
    )


def print_extremes(df: pd.DataFrame) -> None:
    """Print the largest losses and gains for trade-level inspection."""
    display_columns = [
        "ticker_symbol",
        "setup_type",
        "entry_date",
        "exit_date",
        "exit_reason",
        "holding_days",
        "realized_r_multiple",
        "net_pnl_inr",
    ]
    available = [c for c in display_columns if c in df.columns]

    print("\nTOP 10 LOSING TRADES")
    print("-" * 96)
    print(
        df.nsmallest(10, "net_pnl_inr")[available].to_string(
            index=False,
            formatters={
                "realized_r_multiple": "{:+.2f}R".format,
                "net_pnl_inr": "₹{:,.2f}".format,
            },
        )
    )

    print("\nTOP 10 WINNING TRADES")
    print("-" * 96)
    print(
        df.nlargest(10, "net_pnl_inr")[available].to_string(
            index=False,
            formatters={
                "realized_r_multiple": "{:+.2f}R".format,
                "net_pnl_inr": "₹{:,.2f}".format,
            },
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze GANESHA V1 backtest trades.")
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Path to an exported trades.csv. Defaults to the newest backtest artifact.",
    )
    args = parser.parse_args()

    path = args.trades_csv or find_latest_trades_csv()
    if not path.exists():
        raise FileNotFoundError(f"Trade log not found: {path}")

    df = load_trades(path)

    print("=" * 96)
    print("GANESHA V1 — TRADE DIAGNOSTICS")
    print("=" * 96)
    print(f"Trade log: {path}")
    print(f"Total trades: {len(df)}")
    print(f"Net PnL: ₹{df['net_pnl_inr'].sum():,.2f}")
    print(f"Average R: {df['realized_r_multiple'].mean():+.2f}R")
    print(f"Win rate: {(df['net_pnl_inr'] > 0).mean() * 100:.1f}%")
    print(f"Average hold: {df['holding_days'].mean():.1f} sessions")

    print_group_breakdown(df, "setup_type", "BY SETUP")
    print_group_breakdown(df, "exit_reason", "BY EXIT REASON")
    print_group_breakdown(df, "ticker_symbol", "BY STOCK")
    print_extremes(df)

    print("\nDIAGNOSTIC RULE")
    print("-" * 96)
    print("Use this report to identify what to improve before changing entry/exit parameters.")
    print("Do not optimize on the same trades and then treat the result as out-of-sample performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
