"""
GANESHA V1 — Master Agentic AI Orchestrator (Module 10, 14, 27)

ARCHITECTURAL PRINCIPLE:
The Agentic AI (Gemini 2.5 Flash) coordinates the deterministic tools,
synthesizes qualitative market context, enforces the Zero-Compromise rule,
and writes an immutable forensic audit log for every single scan run.

Graceful Degradation Guarantee:
If the Gemini API key is absent, rate-limited, or offline, the orchestrator
automatically falls back to the deterministic rule-based watchlist without crashing:
`STATUS: RULE-BASED WATCHLIST (DEGRADED)`
"""
import json
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from loguru import logger

from src.core.config import settings
from src.audit.audit_logger import AuditTrailLogger
from src.agentic.prompts import GANESHA_SYSTEM_INSTRUCTION, ANALYSIS_PROMPT_TEMPLATE
from src.agentic.tools import (
    get_market_regime,
    check_derivative_expiry,
    check_catb_blackout,
    evaluate_institutional_delivery,
    calculate_risk_coordinates,
    calculate_risk_envelope,
    check_portfolio_circuit_breakers,
    check_correlated_drawdown,
    evaluate_sector_relative_strength,
)
from src.setups.swing_setups import SwingSetupScanner, SwingSetupSignal


class GaneshaAgenticOrchestrator:
    """
    Coordinates Layer 1 (Deterministic Core), Layer 2 (Risk/Calibration),
    and Layer 3 (Gemini 2.5 Flash Agentic Intelligence).
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.api_key = gemini_api_key or settings.gemini_api_key
        self.audit_logger = AuditTrailLogger()
        self._gemini_client = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client if API key is provided."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                for m_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]:
                    try:
                        self._gemini_client = genai.GenerativeModel(
                            model_name=m_name,
                            system_instruction=GANESHA_SYSTEM_INSTRUCTION
                        )
                        logger.info(f"Gemini agentic model ({m_name}) initialized successfully.")
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Could not initialize Gemini model ({e}); running in deterministic mode.")
                self._gemini_client = None
        else:
            logger.info("No Gemini API key detected in config; running in deterministic fallback mode.")

    def run_swing_scan(
        self,
        nifty_history_df,
        candidate_data: Dict[str, Dict[str, Any]],
        trade_date: Optional[date] = None,
        scan_mode: str = "WEEKEND_DEEP_SCAN",
        india_vix: Optional[float] = None,
        active_positions: Optional[List[Dict[str, Any]]] = None,
        recent_closed_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end institutional swing scan.

        Args:
            nifty_history_df: DataFrame of Nifty 50 daily candles.
            candidate_data: Dict mapping ticker_symbol -> {
                                'ohlcv': DataFrame,
                                'delivery_history': List[int],
                                'today_delivery': int,
                                'sector': str
                            }
            trade_date: Scan date (defaults to today).
            scan_mode: "WEEKEND_DEEP_SCAN" or "DAILY_MONITORING".
            india_vix: Latest India VIX reading (optional).

        Returns:
            Dict containing final watchlist, regime status, audit id, and reasoning.
        """
        scan_date = trade_date or date.today()
        trade_date_iso = scan_date.isoformat()
        tool_call_trace: List[Dict[str, Any]] = []

        active_positions = active_positions or []
        recent_closed_trades = recent_closed_trades or []

        logger.info(f"Initiating GANESHA V1 Swing Scan for {trade_date_iso} [{scan_mode}]")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 0: Correlated-Drawdown Pause Brake
        # ═══════════════════════════════════════════════════════════════════════
        drawdown_status = check_correlated_drawdown(recent_closed_trades, scan_date)
        tool_call_trace.append({"tool": "check_correlated_drawdown", "result": drawdown_status})

        if drawdown_status["is_system_paused"]:
            reason = drawdown_status["status_message"]
            logger.warning(f"Scan halted by drawdown circuit breaker: {reason}")

            scan_id = self.audit_logger.log_scan_run(
                scan_type=scan_mode,
                market_regime={},
                expiry_status={},
                approved_candidates=[],
                rejected_candidates=[],
                agentic_reasoning=f"Correlated-Drawdown Pause Active. {reason}",
                tool_call_trace=tool_call_trace,
                execution_status="DRAWDOWN_PAUSE_ACTIVE",
            )
            return {
                "status": "DRAWDOWN_PAUSE_ACTIVE",
                "message": reason,
                "pause_until_date": drawdown_status.get("pause_until_date"),
                "approved_watchlist": [],
                "scan_id": scan_id,
            }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Market Regime Clearance
        # ═══════════════════════════════════════════════════════════════════════
        regime_result = get_market_regime(nifty_history_df, india_vix)
        tool_call_trace.append({"tool": "get_market_regime", "result": regime_result})

        if not regime_result["allow_swing_longs"]:
            reason = regime_result["status_message"]
            logger.warning(f"Scan halted by regime engine: {reason}")

            scan_id = self.audit_logger.log_scan_run(
                scan_type=scan_mode,
                market_regime=regime_result,
                expiry_status={},
                approved_candidates=[],
                rejected_candidates=[],
                agentic_reasoning=f"Broad market conditions prohibitive. {reason}",
                tool_call_trace=tool_call_trace,
                execution_status="REGIME_HALT",
            )
            return {
                "status": "REGIME_HALT",
                "message": reason,
                "regime": regime_result["regime"],
                "approved_watchlist": [],
                "scan_id": scan_id,
            }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Derivative Expiry Calendar Check
        # ═══════════════════════════════════════════════════════════════════════
        expiry_status = check_derivative_expiry(trade_date_iso)
        tool_call_trace.append({"tool": "check_derivative_expiry", "result": expiry_status})
        is_expiry_week = expiry_status["is_expiry_week"]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Setup Screening & Multi-Gate Verification
        # ═══════════════════════════════════════════════════════════════════════
        approved_candidates: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []

        for ticker, data in candidate_data.items():
            df_stock = data.get("ohlcv")
            delivery_history = data.get("delivery_history", [])
            today_delivery = data.get("today_delivery", 0)

            if df_stock is None or len(df_stock) < 30:
                rejected_candidates.append({
                    "ticker": ticker,
                    "reason": "INSUFFICIENT_HISTORICAL_CANDLES_LESS_THAN_30"
                })
                continue

            # Gate A: Corporate Action Timing Buffer (CATB)
            catb_check = check_catb_blackout(ticker, trade_date_iso)
            tool_call_trace.append({"tool": "check_catb_blackout", "ticker": ticker, "result": catb_check})

            if catb_check["is_blackout"]:
                rejected_candidates.append({"ticker": ticker, "reason": catb_check["reason"]})
                continue

            # Gate B: 6 Core Swing Setups Scanner
            signals = SwingSetupScanner.scan_all_setups(
                ticker_symbol=ticker,
                df=df_stock,
                delivery_history=delivery_history,
                is_expiry_week=is_expiry_week
            )

            if not signals:
                rejected_candidates.append({
                    "ticker": ticker,
                    "reason": "NO_CORE_SWING_SETUP_CLASS_MATCHED"
                })
                continue

            # Select highest conviction setup
            top_signal = signals[0]

            # Gate C: Institutional Delivery Volume Expansion Filter
            delivery_check = evaluate_institutional_delivery(
                recent_delivery_history=delivery_history,
                today_delivery=today_delivery,
                is_expiry_week=is_expiry_week
            )
            tool_call_trace.append({"tool": "evaluate_institutional_delivery", "ticker": ticker, "result": delivery_check})

            if not delivery_check["is_institutional_accumulation"]:
                rejected_candidates.append({
                    "ticker": ticker,
                    "setup_found": top_signal.setup_type,
                    "reason": delivery_check["rejection_reason"]
                })
                continue

            # Gate D: Precision Risk Coordinates (+2R Target Verification)
            risk_coords = calculate_risk_coordinates(top_signal.entry_price, top_signal.stop_loss)
            if not risk_coords["is_valid"]:
                rejected_candidates.append({
                    "ticker": ticker,
                    "reason": risk_coords["error"]
                })
                continue

            # Gate E: Friction-Adjusted Position Sizing (1% Risk Rule)
            sizing = calculate_risk_envelope(
                ticker_symbol=ticker,
                entry_price=top_signal.entry_price,
                stop_loss=top_signal.stop_loss,
                target_price=risk_coords["target_price"],
            )
            tool_call_trace.append({"tool": "calculate_risk_envelope", "ticker": ticker, "result": sizing})

            if not sizing["is_executable"]:
                rejected_candidates.append({
                    "ticker": ticker,
                    "setup_found": top_signal.setup_type,
                    "reason": sizing["rejection_reason"]
                })
                continue

            # Gate F: Portfolio Circuit Breakers (Capacity + Sector + Aggregate Risk)
            sector = data.get("sector", "UNKNOWN")
            cb_check = check_portfolio_circuit_breakers(
                active_positions=active_positions + approved_candidates,
                candidate_ticker=ticker,
                candidate_sector=sector,
                candidate_risk_inr=sizing["capital_at_risk_net_inr"],
            )
            tool_call_trace.append({"tool": "check_portfolio_circuit_breakers", "ticker": ticker, "result": cb_check})

            if not cb_check["is_allowed"]:
                rejected_candidates.append({
                    "ticker": ticker,
                    "setup_found": top_signal.setup_type,
                    "reason": cb_check["rejection_reason"]
                })
                continue

            # Gate G: Dual-Layer Mansfield Relative Strength Filter
            sector_series = data.get("sector_series")
            nifty_series = data.get("nifty_series")
            if nifty_series is None and "close" in nifty_history_df.columns:
                nifty_series = nifty_history_df["close"]
            elif nifty_series is None and "Close" in nifty_history_df.columns:
                nifty_series = nifty_history_df["Close"]

            stock_close_series = df_stock["close"] if "close" in df_stock.columns else df_stock["Close"]

            if sector_series is not None and nifty_series is not None:
                rs_check = evaluate_sector_relative_strength(
                    stock_series=stock_close_series,
                    sector_series=sector_series,
                    nifty_series=nifty_series,
                    stock_symbol=ticker,
                    sector_name=sector,
                )
                tool_call_trace.append({"tool": "evaluate_sector_relative_strength", "ticker": ticker, "result": rs_check})

                if not rs_check["passes_rs_filter"]:
                    rejected_candidates.append({
                        "ticker": ticker,
                        "setup_found": top_signal.setup_type,
                        "reason": rs_check["rejection_reason"]
                    })
                    continue

            # Candidate passed ALL deterministic + risk gates!
            candidate_item = {
                "ticker_symbol": ticker,
                "sector": sector,
                "setup_type": top_signal.setup_type,
                "entry_price": risk_coords["entry_price"],
                "stop_loss": risk_coords["stop_loss"],
                "target_price": risk_coords["target_price"],
                "risk_reward_ratio": risk_coords["risk_reward_ratio"],
                "risk_per_share": risk_coords["risk_per_share"],
                "shares_to_buy": sizing["shares_to_buy"],
                "capital_required_inr": sizing["capital_required_inr"],
                "capital_at_risk_net_inr": sizing["capital_at_risk_net_inr"],
                "friction_total_inr": sizing["friction"]["total_friction_inr"],
                "risk_percentage_of_account": sizing["risk_percentage_of_account"],
                "atr_14": top_signal.atr_14,
                "delivery_expansion_ratio": delivery_check["expansion_ratio"],
                "invalidation_level": top_signal.invalidation_level,
                "technical_rationale": top_signal.rationale,
                "is_expiry_week_adjusted": is_expiry_week,
            }
            approved_candidates.append(candidate_item)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Zero-Compromise Check
        # ═══════════════════════════════════════════════════════════════════════
        if not approved_candidates:
            reason = "STATUS: NO VALID SETUP DETECTED — All candidates failed institutional filters."
            logger.info(reason)
            scan_id = self.audit_logger.log_scan_run(
                scan_type=scan_mode,
                market_regime=regime_result,
                expiry_status=expiry_status,
                approved_candidates=[],
                rejected_candidates=rejected_candidates,
                agentic_reasoning=reason,
                tool_call_trace=tool_call_trace,
                execution_status="NO_VALID_SETUP_DETECTED",
            )
            return {
                "status": "NO_VALID_SETUP_DETECTED",
                "message": reason,
                "regime": regime_result["regime"],
                "approved_watchlist": [],
                "rejected_count": len(rejected_candidates),
                "scan_id": scan_id,
            }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Gemini Agentic Orchestration & Explainability
        # ═══════════════════════════════════════════════════════════════════════
        agentic_reasoning = ""
        execution_status = "AI_ANALYSIS_COMPLETE"

        if self._gemini_client:
            try:
                prompt_content = ANALYSIS_PROMPT_TEMPLATE.format(
                    scan_date=trade_date_iso,
                    scan_mode=scan_mode,
                    market_regime=json.dumps(regime_result),
                    expiry_status=json.dumps(expiry_status),
                    num_candidates=len(approved_candidates),
                    candidates_json=json.dumps(approved_candidates, indent=2),
                    rejected_json=json.dumps(rejected_candidates[:10], indent=2),
                )

                response = self._gemini_client.generate_content(prompt_content)
                agentic_reasoning = response.text
                logger.success("Gemini 2.5 Flash qualitative analysis generated successfully.")

            except Exception as e:
                logger.warning(f"Gemini API invocation failed ({e}). Degrading cleanly to deterministic output.")
                execution_status = "RULE_BASED_DEGRADED"
                agentic_reasoning = (
                    "STATUS: RULE-BASED WATCHLIST (DEGRADED) — "
                    "Gemini AI reasoning offline; candidates verified strictly by deterministic quant filters."
                )
        else:
            execution_status = "RULE_BASED_DETERMINISTIC"
            agentic_reasoning = (
                "STATUS: DETERMINISTIC WATCHLIST — "
                "Candidates verified through deterministic regime, CATB, setup, and delivery volume filters."
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: Write Immutable Forensic Audit Record
        # ═══════════════════════════════════════════════════════════════════════
        scan_id = self.audit_logger.log_scan_run(
            scan_type=scan_mode,
            market_regime=regime_result,
            expiry_status=expiry_status,
            approved_candidates=approved_candidates,
            rejected_candidates=rejected_candidates,
            agentic_reasoning=agentic_reasoning,
            tool_call_trace=tool_call_trace,
            execution_status=execution_status,
        )

        return {
            "status": execution_status,
            "scan_id": scan_id,
            "scan_date": trade_date_iso,
            "regime": regime_result["regime"],
            "approved_watchlist": approved_candidates,
            "agentic_reasoning": agentic_reasoning,
            "rejected_count": len(rejected_candidates),
        }
