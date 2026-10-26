"""Audit unresolved entry-prediction outcomes using existing dataset fields.

This is analysis-only. It does not change labels or trading behavior.
Because v1 does not store horizon-end close, this audit uses MFE/MAE to
identify unresolved trades that moved materially in either direction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")


def audit(path: Path) -> None:
    df = pd.read_csv(path)
    required = ["setup_type", "outcome_valid", "mfe_r", "mae_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    unresolved = df[df["outcome_valid"] == 0].copy()
    print("ENTRY_PREDICTION_V1_HORIZON_OUTCOME_AUDIT")
    print(f"rows={len(df)} unresolved={len(unresolved)}")

    if unresolved.empty:
        print("No unresolved outcomes found.")
        return

    for threshold in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
        print(
            f"mfe_ge_{threshold:g}r="
            f"{int((unresolved['mfe_r'] >= threshold).sum())} "
            f"rate={(unresolved['mfe_r'] >= threshold).mean():.4f}"
        )

    for threshold in (-0.25, -0.50, -0.75, -1.00):
        print(
            f"mae_le_{threshold:g}r="
            f"{int((unresolved['mae_r'] <= threshold).sum())} "
            f"rate={(unresolved['mae_r'] <= threshold).mean():.4f}"
        )

    print("\nBY_SETUP")
    by_setup = unresolved.groupby("setup_type").agg(
        rows=("setup_type", "size"),
        unresolved_rate=("outcome_valid", lambda s: 1.0 - float(s.mean())),
        mean_mfe_r=("mfe_r", "mean"),
        median_mfe_r=("mfe_r", "median"),
        mean_mae_r=("mae_r", "mean"),
        median_mae_r=("mae_r", "median"),
    ).sort_values("rows", ascending=False)
    print(by_setup.to_string())

    print("\nINTERPRETATION")
    print("MFE/MAE cannot determine the final horizon-end P&L.")
    print("Use this only to decide whether unresolved outcomes require a new horizon-end-close field.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    audit(args.dataset)


if __name__ == "__main__":
    main()
