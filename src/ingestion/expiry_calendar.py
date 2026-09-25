"""
GANESHA V1 — NSE Derivative Expiry Calendar & Expiry-Week Volatility Filter (Module 16a Upgrade)

REAL-TIME TRADING INTEGRITY:
Because candidate stocks are active in the NSE Futures & Options (F&O) segment,
the final sessions of the monthly derivative expiry cycle (typically ending on the
last Thursday of the month) create massive synthetic volume churn, option pin risk,
and volatility spikes due to physical settlement mandates.

Automated Expiry-Week Volatility Filter:
1. Active Expiry Window: Final 3 trading sessions [Monthly Expiry - 2 trading days, Monthly Expiry]
2. Threshold Tightening (+15%):
   - Delivery Volume SMA multiple tightened from 1.50x to 1.725x
   - Breakout price clearance margin above resistance widened by +15%
   - Required consolidation base duration increased from 15 to 18 sessions
3. F&O Ban / High MWPL Exclusion:
   - Stocks under exchange ban or MWPL utilization >= 85% are strictly blacklisted.
"""
import calendar
from datetime import date, timedelta
from typing import Dict, List, Optional
from loguru import logger


class NSEDerivativeExpiryFilter:
    """
    Manages monthly F&O expiry calendar calculation and enforces expiry-week tightening.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_monthly_expiry_date(year: int, month: int) -> date:
        """
        Calculate the standard NSE monthly equity derivative expiry date:
        Typically the LAST THURSDAY of the month.
        (Note: If that Thursday is a market holiday, it pre-pones to Wednesday).
        """
        # Find the last day of the month
        _, last_day = calendar.monthrange(year, month)
        last_date = date(year, month, last_day)

        # Thursday is weekday 3 (Monday is 0, Sunday is 6)
        days_back = (last_date.weekday() - 3) % 7
        last_thursday = last_date - timedelta(days=days_back)
        return last_thursday

    def is_expiry_week_active(self, current_date: date) -> Dict[str, any]:
        """
        Determines if current_date falls in the final 3 trading sessions of the monthly expiry:
        [Expiry - 2 trading days, Expiry]
        """
        monthly_expiry = self.get_monthly_expiry_date(current_date.year, current_date.month)

        # If current_date is past this month's expiry, check next month's expiry
        if current_date > monthly_expiry:
            next_month = current_date.month + 1 if current_date.month < 12 else 1
            next_year = current_date.year if current_date.month < 12 else current_date.year + 1
            monthly_expiry = self.get_monthly_expiry_date(next_year, next_month)

        # Active window: approximately 4 calendar days covers the 3 trading sessions leading to Thursday
        # e.g., Monday through Thursday
        buffer_start = monthly_expiry - timedelta(days=4)

        is_active = (buffer_start <= current_date <= monthly_expiry)

        return {
            "is_expiry_week": is_active,
            "monthly_expiry_date": monthly_expiry,
            "buffer_start_date": buffer_start,
            "delivery_threshold_multiple": 1.725 if is_active else 1.50,
            "breakout_clearance_multiplier": 1.15 if is_active else 1.00,
            "min_consolidation_sessions": 18 if is_active else 15,
        }

    def check_fno_ban_or_mwpl(
        self,
        ticker_symbol: str,
        current_mwpl_pct: Optional[float] = None,
        is_in_ban: bool = False
    ) -> Dict[str, any]:
        """
        Check if stock is in exchange F&O ban or exceeds 85% MWPL limit.
        """
        if is_in_ban:
            return {
                "is_eligible": False,
                "rejection_reason": f"EXCHANGE_FNO_BAN: {ticker_symbol} is in active F&O ban period.",
                "status_code": "STATUS: DROPPED — F&O BAN ACTIVE"
            }

        if current_mwpl_pct is not None and current_mwpl_pct >= 85.0:
            return {
                "is_eligible": False,
                "rejection_reason": (
                    f"HIGH_MWPL_UTILIZATION: {ticker_symbol} MWPL at {current_mwpl_pct:.1f}% "
                    f"(limit: 85.0%). Extreme synthetic unwinding risk."
                ),
                "status_code": "STATUS: DROPPED — HIGH MWPL RISK"
            }

        return {
            "is_eligible": True,
            "rejection_reason": None,
            "status_code": "FNO_CLEAR"
        }
