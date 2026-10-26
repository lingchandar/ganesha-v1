"""Audit entry-prediction v1 outcome censoring and barrier behavior.

This is analysis-only. It does not change the trading model. It quantifies:
- valid vs unresolved (neither barrier hit) outcomes
- target/stop/same-session barrier events
- setup-specific censoring
- the effect of treating unresolved outcomes as losses vs excluding them
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")


def _metrics(y: pd.Series, score: pd.Series) -> tuple[float, float]:
    if y.nunique() < 2 or score.nunique() < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["trade_date"])
    required = [
        "trade_date", "ticker_symbol", "setup_type", "label_target_before_stop",
        "outcome_valid", "days_to_target", "days_to_stop", "mfe_r", "mae_r",
        "risk_reward_ratio",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.copy()
    df["unresolved"] = df["outcome_valid"] == 0
    df["target_hit"] = df["days_to_target"] > 0
    df["stop_hit"] = df["days_to_stop"] > 0
    df["same_session_barrier"] = df["target_hit"] & df["stop_hit"] & (df["days_to_target"] == df["days_to_stop"])

    print("ENTRY_PREDICTION_V1_LABEL_AUDIT")
    print(f"rows={len(df)}")
    print(f"valid={int(df['outcome_valid'].sum())} unresolved={int(df['unresolved'].sum())} unresolved_rate={df['unresolved'].mean():.4f}")
    print(f"target_hit={int(df['target_hit'].sum())} stop_hit={int(df['stop_hit'].sum())} same_session_barrier={int(df['same_session_barrier'].sum())}")
    print(f"label_positive_all_rows={df['label_target_before_stop'].mean():.4f}")

    print("\nBY_SETUP")
    by_setup = df.groupby("setup_type").agg(
        rows=("setup_type", "size"),
        unresolved=("unresolved", "sum"),
        unresolved_rate=("unresolved", "mean"),
        target_hit=("target_hit", "sum"),
        stop_hit=("stop_hit", "sum"),
        same_session=("same_session_barrier", "sum"),
        positive_rate_all=("label_target_before_stop", "mean"),
    ).sort_values("rows", ascending=False)
    print(by_setup.to_string())

    print("\nOUTCOME_TIMING")
    valid = df[df["outcome_valid"] == 1]
    print(valid[["days_to_target", "days_to_stop", "mfe_r", "mae_r"]].describe().to_string())

    print("\nLABELING_SENSITIVITY")
    # Current production training excludes unresolved outcomes.
    current = df[df["outcome_valid"] == 1]
    # Conservative alternative: unresolved trades are treated as losses.
    all_y = df["label_target_before_stop"].astype(int)
    print(f"current_valid_only rows={len(current)} positive_rate={current['label_target_before_stop'].mean():.4f}")
    print(f"all_rows_unresolved_as_loss rows={len(df)} positive_rate={all_y.mean():.4f}")

    # A diagnostic score using current setup prior, measured on all rows, shows how much
    # much of the apparent signal comes from setup composition rather than features.
    priors = current.groupby("setup_type")["label_target_before_stop"].mean()
    score = df["setup_type"].map(priors).fillna(current["label_target_before_stop"].mean())
    for name, y in [("valid_only", current["label_target_before_stop"]), ("all_rows", all_y)]:
        s = score.loc[current.index] if name == "valid_only" else score
        auc, ap = _metrics(y, s)
        print(f"setup_prior_{name} auc={auc:.4f} ap={ap:.4f}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    audit(args.dataset)


if __name__ == "__main__":
    main()
