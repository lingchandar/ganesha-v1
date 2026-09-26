"""
GANESHA V1 — Friction-Adjusted Position Sizing Engine (Module 10)

SEBI & INSTITUTIONAL EXECUTION REALISM:
Trade sizing is strictly mathematically determined by account equity and
distance to the structural stop-loss, after subtracting all mandatory
Indian statutory friction charges and execution slippage.

The 1% Risk Formula:
    Account Risk Budget = Total Account Capital * 0.01 (1%)
    Friction Costs      = STT + Exchange Charges + SEBI Fee + Stamp Duty + GST + DP Charge + Slippage
    Shares to Buy       = floor( (Account Risk Budget - Flat Charges) / (Risk_Per_Share + Variable_Friction_Per_Share) )

Statutory Cost Schedule (NSE Cash Delivery):
- STT (Securities Transaction Tax): 0.1% on buy turnover + 0.1% on sell turnover
- NSE Exchange Transaction Charge: 0.00297% on total turnover
- SEBI Turnover Fee: ₹10 per crore (0.0001% = 0.000001) on total turnover
- Stamp Duty: 0.015% on buy turnover
- GST: 18% on (Exchange Charges + SEBI Fee + Brokerage)
- DP (Depository) Charge: Flat ₹15.93 per scrip debit on exit (CDSL/NSDL)
- Slippage Buffer: 0.05% on entry + 0.05% on exit (0.10% total)
"""
from dataclasses import dataclass, asdict
import math
from typing import Dict, Optional
from loguru import logger

from src.core.config import settings


@dataclass(frozen=True)
class StatutoryFrictionBreakdown:
    """Detailed breakdown of Indian statutory charges and execution friction."""
    stt_inr: float
    exchange_turnover_inr: float
    sebi_turnover_inr: float
    stamp_duty_inr: float
    gst_inr: float
    dp_charges_inr: float
    slippage_buffer_inr: float
    total_friction_inr: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PositionSizingResult:
    """Full risk envelope and share allocation result."""
    ticker_symbol: str
    entry_price: float
    stop_loss: float
    target_price: float
    risk_per_share: float
    shares_to_buy: int
    capital_required_inr: float
    capital_at_risk_gross_inr: float
    capital_at_risk_net_inr: float
    account_risk_budget_inr: float
    friction: StatutoryFrictionBreakdown
    risk_percentage_of_account: float
    is_executable: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        res = asdict(self)
        res["friction"] = self.friction.to_dict()
        return res


class PositionSizingEngine:
    """
    Calculates deterministic, friction-buffered position sizes for NSE equity swing trades.
    """

    # Statutory delivery tax rates
    STT_BUY_RATE = 0.001          # 0.1%
    STT_SELL_RATE = 0.001         # 0.1%
    NSE_EXCHANGE_RATE = 0.0000297 # 0.00297%
    SEBI_RATE = 0.000001          # ₹10 per crore (0.0001%)
    STAMP_DUTY_RATE = 0.00015     # 0.015% (Buy only)
    GST_RATE = 0.18               # 18% on exchange charges + SEBI fees + brokerage
    FLAT_DP_CHARGE = 15.93        # Flat ₹15.93 per scrip sell debit
    SLIPPAGE_RATE_PER_SIDE = 0.0005 # 0.05% entry + 0.05% exit

    # Cash Swing Trading Risk Ceiling: Maximum 25% capital in any single position
    MAX_POSITION_ALLOCATION_PCT = 0.25

    @classmethod
    def calculate_statutory_friction(
        cls,
        shares: int,
        entry_price: float,
        exit_price: float,
        brokerage_per_order: float = 0.0
    ) -> StatutoryFrictionBreakdown:
        """
        Computes the complete, exact Indian statutory delivery taxes and slippage for a completed trade.
        """
        if shares <= 0:
            return StatutoryFrictionBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        buy_turnover = shares * entry_price
        sell_turnover = shares * exit_price
        total_turnover = buy_turnover + sell_turnover

        stt = round((buy_turnover * cls.STT_BUY_RATE) + (sell_turnover * cls.STT_SELL_RATE), 2)
        exchange_fee = round(total_turnover * cls.NSE_EXCHANGE_RATE, 2)
        sebi_fee = round(total_turnover * cls.SEBI_RATE, 2)
        stamp_duty = round(buy_turnover * cls.STAMP_DUTY_RATE, 2)
        
        # Total brokerage (Buy + Sell, usually ₹0 for delivery or flat ₹20)
        total_brokerage = brokerage_per_order * 2.0
        taxable_services = exchange_fee + sebi_fee + total_brokerage
        gst = round(taxable_services * cls.GST_RATE, 2)

        dp_charges = cls.FLAT_DP_CHARGE
        slippage = round((buy_turnover * cls.SLIPPAGE_RATE_PER_SIDE) + (sell_turnover * cls.SLIPPAGE_RATE_PER_SIDE), 2)

        total_friction = round(stt + exchange_fee + sebi_fee + stamp_duty + gst + dp_charges + slippage, 2)

        return StatutoryFrictionBreakdown(
            stt_inr=stt,
            exchange_turnover_inr=exchange_fee,
            sebi_turnover_inr=sebi_fee,
            stamp_duty_inr=stamp_duty,
            gst_inr=gst,
            dp_charges_inr=dp_charges,
            slippage_buffer_inr=slippage,
            total_friction_inr=total_friction,
        )

    @classmethod
    def calculate_position_size(
        cls,
        ticker_symbol: str,
        entry_price: float,
        stop_loss: float,
        target_price: Optional[float] = None,
        total_capital: Optional[float] = None,
        max_risk_pct: Optional[float] = None,
        brokerage_per_order: float = 0.0,
    ) -> PositionSizingResult:
        """
        Calculates exact shares to buy such that (Gross Loss + Complete Friction) <= 1% Account Risk Budget.

        Args:
            ticker_symbol: NSE ticker symbol (e.g. "RELIANCE")
            entry_price: Recommended entry price in INR
            stop_loss: Structural stop-loss price in INR
            target_price: Profit target price (defaults to +2R)
            total_capital: Account capital in INR (defaults to config: 500,000)
            max_risk_pct: Maximum risk fraction (defaults to config: 0.01)
            brokerage_per_order: Delivery brokerage per order (default 0.0)

        Returns:
            PositionSizingResult with exact share count, capital required, and friction breakdown.
        """
        capital = total_capital or settings.total_capital_inr
        risk_pct = max_risk_pct or settings.max_risk_per_trade_pct

        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return PositionSizingResult(
                ticker_symbol=ticker_symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price or entry_price,
                risk_per_share=risk_per_share,
                shares_to_buy=0,
                capital_required_inr=0.0,
                capital_at_risk_gross_inr=0.0,
                capital_at_risk_net_inr=0.0,
                account_risk_budget_inr=round(capital * risk_pct, 2),
                friction=StatutoryFrictionBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
                risk_percentage_of_account=0.0,
                is_executable=False,
                rejection_reason=f"INVALID_STOP_LOSS: Entry price (₹{entry_price:.2f}) must exceed stop-loss (₹{stop_loss:.2f})."
            )

        if target_price is None:
            target_price = round(entry_price + (2.0 * risk_per_share), 2)

        risk_budget = capital * risk_pct
        flat_charges = cls.FLAT_DP_CHARGE + (brokerage_per_order * 2.0 * (1.0 + cls.GST_RATE))

        # Check if even flat charges exceed risk budget
        if flat_charges >= risk_budget:
            return PositionSizingResult(
                ticker_symbol=ticker_symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                risk_per_share=risk_per_share,
                shares_to_buy=0,
                capital_required_inr=0.0,
                capital_at_risk_gross_inr=0.0,
                capital_at_risk_net_inr=0.0,
                account_risk_budget_inr=round(risk_budget, 2),
                friction=StatutoryFrictionBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
                risk_percentage_of_account=0.0,
                is_executable=False,
                rejection_reason=f"RISK_BUDGET_TOO_SMALL: Risk budget (₹{risk_budget:.2f}) cannot cover flat fees (₹{flat_charges:.2f})."
            )

        # Variable friction per share at the stop-loss exit point (worst-case risk realization)
        buy_var_rate = (
            cls.STT_BUY_RATE
            + cls.STAMP_DUTY_RATE
            + cls.SLIPPAGE_RATE_PER_SIDE
            + (cls.NSE_EXCHANGE_RATE + cls.SEBI_RATE) * (1.0 + cls.GST_RATE)
        )
        sell_var_rate = (
            cls.STT_SELL_RATE
            + cls.SLIPPAGE_RATE_PER_SIDE
            + (cls.NSE_EXCHANGE_RATE + cls.SEBI_RATE) * (1.0 + cls.GST_RATE)
        )
        variable_friction_per_share = (entry_price * buy_var_rate) + (stop_loss * sell_var_rate)

        # Net effective risk per share including variable friction
        effective_risk_per_share = risk_per_share + variable_friction_per_share

        # Mathematical floor allocation
        raw_shares = math.floor((risk_budget - flat_charges) / effective_risk_per_share)

        if raw_shares <= 0:
            return PositionSizingResult(
                ticker_symbol=ticker_symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                risk_per_share=round(risk_per_share, 2),
                shares_to_buy=0,
                capital_required_inr=0.0,
                capital_at_risk_gross_inr=0.0,
                capital_at_risk_net_inr=0.0,
                account_risk_budget_inr=round(risk_budget, 2),
                friction=StatutoryFrictionBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
                risk_percentage_of_account=0.0,
                is_executable=False,
                rejection_reason=f"CAPITAL_RISK_BREACH: 1 share risk (₹{effective_risk_per_share:.2f}) exceeds 1% budget (₹{risk_budget:.2f})."
            )

        # Cap by maximum portfolio allocation (no single position may exceed 25% of total capital)
        max_capital_for_trade = capital * cls.MAX_POSITION_ALLOCATION_PCT
        shares_by_capital_cap = math.floor(max_capital_for_trade / entry_price)
        final_shares = min(raw_shares, shares_by_capital_cap)

        if final_shares <= 0:
            return PositionSizingResult(
                ticker_symbol=ticker_symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                risk_per_share=round(risk_per_share, 2),
                shares_to_buy=0,
                capital_required_inr=0.0,
                capital_at_risk_gross_inr=0.0,
                capital_at_risk_net_inr=0.0,
                account_risk_budget_inr=round(risk_budget, 2),
                friction=StatutoryFrictionBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
                risk_percentage_of_account=0.0,
                is_executable=False,
                rejection_reason=f"POSITION_SIZE_CAP_BREACH: Capital required for 1 share (₹{entry_price:.2f}) exceeds 25% position cap (₹{max_capital_for_trade:.2f})."
            )

        # Compute exact final friction at stop-loss realization
        friction = cls.calculate_statutory_friction(
            shares=final_shares,
            entry_price=entry_price,
            exit_price=stop_loss,
            brokerage_per_order=brokerage_per_order,
        )

        gross_risk = round(final_shares * risk_per_share, 2)
        net_risk = round(gross_risk + friction.total_friction_inr, 2)
        capital_required = round(final_shares * entry_price, 2)
        actual_risk_pct = round((net_risk / capital) * 100.0, 3)

        return PositionSizingResult(
            ticker_symbol=ticker_symbol,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_price=round(target_price, 2),
            risk_per_share=round(risk_per_share, 2),
            shares_to_buy=final_shares,
            capital_required_inr=capital_required,
            capital_at_risk_gross_inr=gross_risk,
            capital_at_risk_net_inr=net_risk,
            account_risk_budget_inr=round(risk_budget, 2),
            friction=friction,
            risk_percentage_of_account=actual_risk_pct,
            is_executable=True,
            rejection_reason=None,
        )
