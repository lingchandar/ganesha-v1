"""Analyze economic outcomes of unresolved entry-prediction trades."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "trade_date",
    "ticker_symbol",
    "setup_type",
    "outcome_valid",
    "horizon_end_r",
]

BUCKET_BINS = [-float("inf"), -0.50, 0.0, 0.25, 0.50, 1.00, float("inf")]
BUCKET_LABELS = [
    "<-0.50R",
    "-0.50R_to_0R",
    "0R_to_0.25R",
    "0.25R_to_0.50R",
    "0.50R_to_1R",
    ">=1R",
]


def _validate(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _summary(df: pd.DataFrame) -> dict[str, float]:
    r = pd.to_numeric(df["horizon_end_r"], errors="coerce").dropna()
    if r.empty:
        return {
            "rows": 0,
            "mean_r": float("nan"),
            "median_r": float("nan"),
            "positive_rate": float("nan"),
            "ge_0_25r": float("nan"),
            "ge_0_50r": float("nan"),
            "ge_1r": float("nan"),
            "le_neg_0_25r": float("nan"),
            "le_neg_0_50r": float("nan"),
        }
    return {
        "rows": float(len(r)),
        "mean_r": float(r.mean()),
        "median_r": float(r.median()),
        "positive_rate": float((r > 0).mean()),
        "ge_0_25r": float((r >= 0.25).mean()),
        "ge_0_50r": float((r >= 0.50).mean()),
        "ge_1r": float((r >= 1.00).mean()),
        "le_neg_0_25r": float((r <= -0.25).mean()),
        "le_neg_0_50r": float((r <= -0.50).mean()),
    }


def _print_summary(name: str, df: pd.DataFrame) -> None:
    s = _summary(df)
    print(name)
    print(f"rows={int(s['rows'])}")
    print(f"mean_r={s['mean_r']:.4f} median_r={s['median_r']:.4f}")
    print(f"positive_rate={s['positive_rate']:.4f}")
    print(f"ge_0.25r={s['ge_0_25r']:.4f} ge_0.50r={s['ge_0_50r']:.4f} ge_1r={s['ge_1r']:.4f}")
    print(f"le_-0.25r={s['le_neg_0_25r']:.4f} le_-0.50r={s['le_neg_0_50r']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/prediction/entry_prediction_dataset_v1.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    _validate(df)
    df["horizon_end_r"] = pd.to_numeric(df["horizon_end_r"], errors="coerce")
    unresolved = df[df["outcome_valid"] == 0].copy()
    unresolved = unresolved.dropna(subset=["horizon_end_r"])

    print("ENTRY_PREDICTION_V1_TIMEOUT_DISTRIBUTION")
    print(f"dataset_rows={len(df)} unresolved_rows={len(unresolved)} unresolved_rate={len(unresolved) / len(df):.4f}")
    print()
    _print_summary("OVERALL_TIMEOUTS", unresolved)

    unresolved["horizon_bucket"] = pd.cut(
        unresolved["horizon_end_r"], bins=BUCKET_BINS, labels=BUCKET_LABELS, right=False
    )
    print("\nHORIZON_END_R_BUCKETS")
    buckets = unresolved["horizon_bucket"].value_counts(sort=False)
    bucket_frame = pd.DataFrame({"rows": buckets, "rate": buckets / len(unresolved)})
    print(bucket_frame.to_string())

    print("\nTIMEOUTS_BY_SETUP")
    setup = unresolved.groupby("setup_type", sort=False).agg(
        rows=("horizon_end_r", "size"),
        mean_r=("horizon_end_r", "mean"),
        median_r=("horizon_end_r", "median"),
        positive_rate=("horizon_end_r", lambda s: (s > 0).mean()),
        ge_0_25r=("horizon_end_r", lambda s: (s >= 0.25).mean()),
        ge_0_50r=("horizon_end_r", lambda s: (s >= 0.50).mean()),
        ge_1r=("horizon_end_r", lambda s: (s >= 1.00).mean()),
        le_neg_0_50r=("horizon_end_r", lambda s: (s <= -0.50).mean()),
    ).sort_values("rows", ascending=False)
    print(setup.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nTIMEOUTS_BY_YEAR")
    yearly = unresolved.assign(year=pd.to_datetime(unresolved["trade_date"]).dt.year).groupby("year").agg(
        rows=("horizon_end_r", "size"),
        mean_r=("horizon_end_r", "mean"),
        median_r=("horizon_end_r", "median"),
        positive_rate=("horizon_end_r", lambda s: (s > 0).mean()),
        ge_0_50r=("horizon_end_r", lambda s: (s >= 0.50).mean()),
    )
    print(yearly.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nINTERPRETATION")
    print("horizon_end_r is price-based R at the 15-session horizon, before transaction costs.")
    print("Do not treat timeout outcomes as automatic losses or as realized trade P&L without execution-cost modeling.")
    print("Use the distribution above to choose a label policy, then validate that policy walk-forward before retraining.")


if __name__ == "__main__":
    main()
