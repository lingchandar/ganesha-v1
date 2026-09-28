"""
GANESHA V1 — Local FastAPI Dashboard & Monitoring Server (Module 8, 14, 26)

SEBI COMPLIANCE & HUMAN-IN-THE-LOOP POSTURE:
This dashboard serves as the human-in-the-loop manual decision support console.
It presents verified algorithmic swing signals, dynamic stop-losses, and +2R targets.
Zero auto-execution logic is present. All order entries and exits are executed
manually by the user via their broker terminal.
"""
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from loguru import logger

from src.core.config import settings
from src.audit.audit_logger import AuditTrailLogger
from src.agentic.orchestrator import GaneshaAgenticOrchestrator
from src.risk.exit_engine import PrecisionExitEngine


app = FastAPI(
    title="🕉️ GANESHA V1 — Swing Trading Agentic AI",
    description="Real-Time Swing Trading Signal Engine & Decision Support System",
    version="1.0.0"
)

# Template & Static directories
DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Audit trail logger instance
audit_logger = AuditTrailLogger()


class ExitCheckRequest(BaseModel):
    entry_price: float
    stop_loss: float
    target_price: float
    invalidation_level: float
    days_held: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    has_upcoming_earnings: bool = False


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Render the primary swing trading cockpit UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "engine_title": "GANESHA V1",
            "version": "1.0.0 (Production)",
            "compliance_status": "SEBI Signal-Only Compliant",
        }
    )


@app.get("/api/status")
async def get_system_status():
    """Returns current system health, regime indicators, and database status."""
    recent_scans = audit_logger.list_recent_scans(limit=1)
    last_scan = recent_scans[0] if recent_scans else None

    return {
        "status": "ONLINE",
        "engine": "GANESHA V1 Institutional Swing Engine",
        "broker_configured": settings.broker_name,
        "database_connected": True,
        "last_scan": last_scan,
        "current_time_ist": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/watchlist")
async def get_latest_watchlist():
    """Returns the latest approved swing watchlist from the forensic audit records."""
    recent_scans = audit_logger.list_recent_scans(limit=1)
    if not recent_scans:
        return {
            "status": "NO_SCAN_DATA",
            "message": "No scan records recorded yet. Trigger a scan once market data is connected.",
            "watchlist": [],
        }

    # Load the latest scan JSON file
    latest_file = Path(recent_scans[0]["file_path"])
    try:
        import json
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "status": data.get("metadata", {}).get("execution_status"),
            "scan_id": data.get("metadata", {}).get("scan_id"),
            "timestamp": data.get("metadata", {}).get("timestamp_utc"),
            "market_regime": data.get("market_regime", {}),
            "watchlist": data.get("approved_watchlist", []),
            "agentic_reasoning": data.get("agentic_ai_orchestration", {}).get("reasoning_summary", ""),
        }
    except Exception as e:
        logger.error(f"Error reading scan file {latest_file}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit-logs")
async def get_audit_logs():
    """Returns list of recent immutable scan records for forensic audit."""
    scans = audit_logger.list_recent_scans(limit=20)
    return {"records": scans}


@app.post("/api/exit/evaluate")
async def evaluate_trade_exit(req: ExitCheckRequest):
    """
    Evaluates an active swing position against the 5 real-time exit rules.
    """
    candle = {
        "open": req.open_price,
        "high": req.high_price,
        "low": req.low_price,
        "close": req.close_price,
    }

    try:
        res = PrecisionExitEngine.evaluate_active_trade(
            entry_price=req.entry_price,
            stop_loss=req.stop_loss,
            target_price=req.target_price,
            invalidation_level=req.invalidation_level,
            days_held=req.days_held,
            daily_candle=candle,
            has_upcoming_earnings=req.has_upcoming_earnings,
        )
        return {
            "is_exit_triggered": res.is_exit_triggered,
            "exit_reason": res.exit_reason,
            "exit_price": res.exit_price,
            "realized_r_multiple": res.realized_r_multiple,
            "holding_days_elapsed": res.holding_days_elapsed,
            "pnl_percentage": res.pnl_percentage,
            "diagnostic_notes": res.diagnostic_notes,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
