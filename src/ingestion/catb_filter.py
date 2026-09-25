"""
GANESHA V1 — Corporate Action Timing Buffer (CATB) Filter (Module 3)

REAL-TIME TRADING INTEGRITY:
Corporate actions (stock splits, bonus shares, rights issues) create artificial
overnight price drops that uncalibrated technical scanners misinterpret as massive
breakdowns or sudden false volume/price breakouts.

The 5-Day Trading Blackout Rule:
    Blackout Window = [Ex-Date - 3 trading days, Ex-Date + 2 trading days]

Any stock entering or residing within this window is immediately dropped
from the active screening and swing watchlist to protect real capital.
"""
from datetime import date, timedelta
from typing import Dict, List, Optional
from sqlalchemy import text
from loguru import logger

from src.core.database import get_db_session


class CorporateActionTimingBuffer:
    """
    Evaluates corporate action proximity and enforces the 5-day trading blackout rule.
    """

    def __init__(self, pre_ex_days: int = 3, post_ex_days: int = 2):
        self.pre_ex_days = pre_ex_days
        self.post_ex_days = post_ex_days

    def is_in_blackout(
        self,
        ticker_symbol: str,
        current_date: date,
        upcoming_actions: Optional[List[Dict]] = None,
    ) -> Dict[str, any]:
        """
        Check whether a stock is inside the CATB Blackout Window.
        
        Window = [Ex-Date - pre_ex_days trading sessions, Ex-Date + post_ex_days trading sessions]
        Using calendar days approximation: [Ex-Date - 5 calendar days, Ex-Date + 3 calendar days]
        """
        if upcoming_actions is None:
            upcoming_actions = self.get_corporate_actions_for_symbol(ticker_symbol)

        for action in upcoming_actions:
            ex_date = action["ex_date"]
            if isinstance(ex_date, str):
                ex_date = date.fromisoformat(ex_date)

            # Define blackout bounds (approx 3 trading days ~ 5 cal days, 2 trading days ~ 3 cal days)
            blackout_start = ex_date - timedelta(days=5)
            blackout_end = ex_date + timedelta(days=3)

            if blackout_start <= current_date <= blackout_end:
                reason = (
                    f"CATB_BLACKOUT_ACTIVE: Stock in corporate action blackout window "
                    f"[{blackout_start} to {blackout_end}] for {action.get('action_type', 'ACTION')} "
                    f"on Ex-Date {ex_date}"
                )
                logger.warning(f"Ticker {ticker_symbol} rejected: {reason}")
                return {
                    "is_blackout": True,
                    "action_type": action.get("action_type"),
                    "ex_date": ex_date,
                    "reason": reason,
                    "status_code": "STATUS: DROPPED — CATB BLACKOUT ACTIVE",
                }

        return {
            "is_blackout": False,
            "action_type": None,
            "ex_date": None,
            "reason": None,
            "status_code": "CATB_CLEAR",
        }

    def get_corporate_actions_for_symbol(self, ticker_symbol: str) -> List[Dict]:
        """
        Query database corporate_actions_calendar table for active upcoming/recent actions.
        """
        try:
            with get_db_session() as db:
                rows = db.execute(
                    text("""
                        SELECT ticker_symbol, action_type, ex_date, record_date, adjustment_factor
                        FROM corporate_actions_calendar
                        WHERE ticker_symbol = :ticker
                        ORDER BY ex_date ASC
                    """),
                    {"ticker": ticker_symbol}
                ).fetchall()

                return [
                    {
                        "ticker_symbol": row[0],
                        "action_type": row[1],
                        "ex_date": row[2],
                        "record_date": row[3],
                        "adjustment_factor": float(row[4]) if row[4] else 1.0,
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error querying corporate actions for {ticker_symbol}: {e}")
            return []
