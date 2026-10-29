"""Append-friendly outcome ledger schema for V2 research and shadow runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional
import json


@dataclass(frozen=True)
class OutcomeRecord:
    ticker_symbol: str
    setup_type: str
    setup_date: date
    entry_date: Optional[date] = None
    exit_date: Optional[date] = None
    state: str = "DISCOVERED"
    thesis_score: Optional[float] = None
    prediction_setup_probability: Optional[float] = None
    prediction_entry_probability: Optional[float] = None
    prediction_t1_probability: Optional[float] = None
    prediction_t2_probability: Optional[float] = None
    prediction_1r_probability: Optional[float] = None
    prediction_2r_probability: Optional[float] = None
    prediction_3r_probability: Optional[float] = None
    expected_r: Optional[float] = None
    realized_r: Optional[float] = None
    holding_days: Optional[int] = None
    mfe_r: Optional[float] = None
    mae_r: Optional[float] = None
    exit_reason: Optional[str] = None
    earnings_state: Optional[str] = None
    analogue_count: Optional[int] = None
    outcome_status: str = "PENDING"
    model_version: Optional[str] = None
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("setup_date", "entry_date", "exit_date"):
            if value[key] is not None:
                value[key] = value[key].isoformat()
        value["reason_codes"] = list(self.reason_codes)
        return value


def append_jsonl(path: Path | str, record: OutcomeRecord) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
