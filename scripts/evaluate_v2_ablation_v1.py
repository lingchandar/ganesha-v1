"""Report the V2 ablation ladder from an outcome ledger.

The runner is intentionally conservative: it reports available rows and does
not infer unavailable outcomes or promote any component.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.validation.evaluation_metrics import strategy_metrics


def _metrics(frame: pd.DataFrame, name: str) -> dict:
    if frame.empty or "realized_r" not in frame:
        return {"name": name, "trade_count": 0}
    valid = frame[pd.to_numeric(frame["realized_r"], errors="coerce").notna()].copy()
    if valid.empty:
        return {"name": name, "trade_count": 0}
    reasons = valid.get("exit_reason", pd.Series([""] * len(valid))).astype(str).tolist()
    result = strategy_metrics(
        pd.to_numeric(valid["realized_r"], errors="coerce").tolist(),
        pd.to_numeric(valid.get("holding_days", pd.Series([0] * len(valid))), errors="coerce").fillna(0).tolist(),
        pd.to_numeric(valid.get("mfe_r", pd.Series([0] * len(valid))), errors="coerce").fillna(0).tolist(),
        pd.to_numeric(valid.get("mae_r", pd.Series([0] * len(valid))), errors="coerce").fillna(0).tolist(),
        coverage=len(valid) / max(len(frame), 1),
        exit_reasons=reasons,
    )
    return {"name": name, **result.__dict__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("data/prediction/v2_outcome_ledger.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/prediction/v2_ablation_report.csv"))
    args = parser.parse_args()
    if not args.ledger.exists():
        print(f"LEDGER_NOT_FOUND: {args.ledger}")
        return 2
    frame = pd.read_json(args.ledger, lines=True)
    stages = [("baseline", frame)]
    if "prediction_entry_probability" in frame:
        stages.append(("+candidate_prediction", frame[frame["prediction_entry_probability"].fillna(0) >= 0.20]))
    if "thesis_score" in frame:
        stages.append(("+thesis_strength", frame[frame["thesis_score"].fillna(0) >= 50]))
    if "earnings_state" in frame:
        stages.append(("+earnings_context", frame[frame["earnings_state"].notna()]))
    report = pd.DataFrame([_metrics(stage, name) for name, stage in stages])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(report.to_string(index=False))
    print(f"REPORT: {args.output}")
    print("NOTE: ablation output is diagnostic. It does not promote production components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
