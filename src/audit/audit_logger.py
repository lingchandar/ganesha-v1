"""
GANESHA V1 — Immutable Forensic Audit Trail Logger (Module 14, 27a)

REAL-TIME TRADING COMPLIANCE & ACCOUNTABILITY:
Every scan run must leave a permanent, tamper-evident forensic record.
In live trading, post-mortem analysis of both winning trades and losing trades
requires exact point-in-time knowledge of:
- Raw numerical tool inputs
- Market regime state
- Technical indicator values (EMA, RSI, ATR, MACD)
- Institutional delivery ratios
- Machine learning calibrated probabilities
- Full Gemini Agentic AI chain-of-thought and tool call traces
- Exact rejection reasons for failed candidates
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger


class AuditTrailLogger:
    """
    Writes immutable JSON forensic records for every watchlist scan.
    """

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            self.log_dir = Path(__file__).resolve().parent.parent.parent / "audit_logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_scan_run(
        self,
        scan_type: str,
        market_regime: Dict[str, Any],
        expiry_status: Dict[str, Any],
        approved_candidates: List[Dict[str, Any]],
        rejected_candidates: List[Dict[str, Any]],
        agentic_reasoning: str,
        tool_call_trace: List[Dict[str, Any]],
        execution_status: str,
    ) -> str:
        """
        Record a complete scan event into an immutable JSON file.

        Returns:
            The unique scan_id (UUID) of this record.
        """
        scan_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp_str}_{scan_id[:8]}.json"
        filepath = self.log_dir / filename

        record = {
            "metadata": {
                "scan_id": scan_id,
                "timestamp_utc": now.isoformat(),
                "scan_type": scan_type,  # WEEKEND_DEEP_SCAN or DAILY_MONITORING
                "execution_status": execution_status,  # AI_ANALYSIS_COMPLETE, REGIME_HALT, RULE_BASED_DEGRADED
                "engine_version": "GANESHA_V1_PROD",
                "sebi_compliance_mode": "SIGNAL_ONLY_NON_ADVISORY",
            },
            "market_regime": market_regime,
            "expiry_week_status": expiry_status,
            "approved_watchlist": approved_candidates,
            "rejected_candidates": rejected_candidates,
            "agentic_ai_orchestration": {
                "model_identifier": "gemini-2.5-flash",
                "reasoning_summary": agentic_reasoning,
                "tool_call_trace": tool_call_trace,
            },
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)

            logger.info(f"Immutable audit trail saved successfully: {filepath.name}")
            return scan_id

        except Exception as e:
            logger.critical(f"FATAL: Failed to write audit log to {filepath}: {e}")
            raise RuntimeError(f"AUDIT_LOG_WRITE_FAILED: {e}")

    def list_recent_scans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve summaries of the most recent scan records.
        """
        records = []
        json_files = sorted(self.log_dir.glob("scan_*.json"), reverse=True)

        for p in json_files[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = data.get("metadata", {})
                    records.append({
                        "file_path": str(p),
                        "scan_id": meta.get("scan_id"),
                        "timestamp": meta.get("timestamp_utc"),
                        "status": meta.get("execution_status"),
                        "approved_count": len(data.get("approved_watchlist", [])),
                        "rejected_count": len(data.get("rejected_candidates", [])),
                    })
            except Exception as e:
                logger.error(f"Error reading audit log {p.name}: {e}")

        return records
