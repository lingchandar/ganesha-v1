"""
GANESHA V1 — Historical Backtesting Engine (Stage 9)

Simulates institutional swing trading over authentic multi-year NSE daily candles.
Guarantees:
1. Zero look-ahead bias: Evening scans strictly use data up to Day T close.
2. Realistic order execution: GTT limit fills on Day T+1, gap-down fills at actual open.
3. Statutory Indian delivery transaction friction on all entries and exits.
4. Comprehensive portfolio risk controls: 1% risk sizing, capacity limits, sector caps,
   and 7-day correlated drawdown halt.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger

from src.core.config import settings
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.performance import (
    BacktestTrade,
    DailyEquityPoint,
    PerformanceSummary,
    BacktestPerformanceAnalyzer,
)
from src.regime.market_regime import MarketRegimeEngine
from src.regime.sector_strength import RelativeStrengthEngine
from src.setups.swing_setups import SwingSetupScanner
from src.risk.exit_engine import PrecisionExitEngine, ExitReason
from src.risk.position_sizer import PositionSizingEngine
from src.risk.circuit_breakers import PortfolioCircuitBreakers
from src.ingestion.nse_delivery_client import NSEArchiveDeliveryClient
from src.ingestion.expiry_calendar import NSEDerivativeExpiryFilter


@dataclass
class BacktestConfig:
    """Configurable backtest simulation parameters."""
    initial_capital_inr: float = 500_000.0
    max_risk_pct: float = 0.01  # 1.0% risk per trade
    max_concurrent_positions: int = 5
    max_per_sector: int = 2
    min_reward_risk: float = 2.0  # Minimum +2R target
    enable_friction: bool = True
    enable_circuit_breakers: bool = True
    enable_drawdown_brake: bool = True
    enable_regime_filter: bool = True
    enable_rs_filter: bool = True
    min_warmup_bars: int = 300
    start_date: Optional[date] = None
    end_date: Optional[date] = None


@dataclass
class PendingOrder:
    """Pending GTT limit order generated at Day T close to be executed on Day T+1."""
    ticker_symbol: str
    sector: str
    setup_type: str
    entry_price: float
    stop_loss: float
    target_price: float
    invalidation_level: float
    shares: int
    capital_required_inr: float
    capital_at_risk_net_inr: float
    estimated_entry_friction_inr: float
    risk_reward_ratio: float
    scan_date: date


@dataclass
class BacktestPosition:
    """Active open swing position in portfolio."""
    position_id: str
    ticker_symbol: str
    sector: str
    setup_type: str
    entry_date: date
    entry_price: float
    shares: int
    capital_invested: float
    initial_stop_loss: float
    target_price: float
    invalidation_level: float
    entry_friction_inr: float
    capital_at_risk_net_inr: float
    days_held: int = 0


@dataclass
class BacktestResult:
    """Complete results of a backtest run."""
    summary: PerformanceSummary
    trades: List[BacktestTrade]
    equity_curve: List[DailyEquityPoint]
    config: BacktestConfig


class BacktestEngine:
    """
    Master bar-by-bar backtest simulation engine.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        data_loader: Optional[HistoricalDataLoader] = None,
    ):
        self.config = config or BacktestConfig()
        self.data_loader = data_loader or HistoricalDataLoader()
        self.expiry_filter = NSEDerivativeExpiryFilter()

        # State tracking
        self.cash: float = self.config.initial_capital_inr
        self.active_positions: List[BacktestPosition] = []
        self.pending_orders: List[PendingOrder] = []
        self.closed_trades: List[BacktestTrade] = []
        self.daily_equity_curve: List[DailyEquityPoint] = []
        self.peak_equity: float = self.config.initial_capital_inr
        self.system_paused_until: Optional[date] = None

    def run(self) -> BacktestResult:
        """
        Executes complete historical backtest across all available trading sessions.
        """
        # Ensure data is loaded
        if not self.data_loader._is_loaded:
            self.data_loader.load_data(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

        trading_dates = self.data_loader.trading_dates
        if len(trading_dates) <= self.config.min_warmup_bars:
            raise ValueError(
                f"Insufficient historical data ({len(trading_dates)} days) "
                f"for warmup requirement of {self.config.min_warmup_bars} days."
            )

        warmup_dates = trading_dates[: self.config.min_warmup_bars]
        simulation_dates = trading_dates[self.config.min_warmup_bars :]

        logger.info(
            f"Starting backtest simulation: {len(simulation_dates)} trading sessions "
            f"({simulation_dates[0]} to {simulation_dates[-1]}) with "
            f"₹{self.config.initial_capital_inr:,.2f} initial capital."
        )

        for current_date in simulation_dates:
            self._step_simulation_day(current_date)

        # Force-close any open positions on the final day for clean accounting
        if simulation_dates and self.active_positions:
            final_date = simulation_dates[-1]
            self._close_all_open_positions(final_date, reason="BACKTEST_TERMINATION")

        # Compile final performance summary
        summary = BacktestPerformanceAnalyzer.analyze(
            trades=self.closed_trades,
            equity_curve=self.daily_equity_curve,
            initial_capital=self.config.initial_capital_inr,
        )

        logger.success(
            f"Backtest complete: {summary.total_trades} trades executed. "
            f"Win Rate: {summary.win_rate_pct:.1f}%, Profit Factor: {summary.profit_factor:.2f}, "
            f"Net PnL: ₹{summary.net_profit_inr:,.2f} ({summary.net_return_pct:.2f}%), "
            f"Max DD: {summary.max_drawdown_pct:.2f}%."
        )

        return BacktestResult(
            summary=summary,
            trades=self.closed_trades,
            equity_curve=self.daily_equity_curve,
            config=self.config,
        )

    def _step_simulation_day(self, current_date: date) -> None:
        """
        Processes a single trading day in strict chronological order:
        1. Morning/Intraday: Update existing positions & check exits.
        2. Morning: Execute pending entry orders generated from Day T-1 close.
        3. End of Day: Mark-to-market valuation, drawdown brake check, record equity snapshot.
        4. Evening: Screen candidates using data strictly up to Day T close and place pending orders for Day T+1.
        """
        # ── 1. Process Exits for Open Positions ──
        self._evaluate_active_position_exits(current_date)

        # ── 2. Process Pending Entry Orders ──
        self._execute_pending_orders(current_date)

        # ── 3. End of Day Mark-to-Market & Portfolio Valuation ──
        self._record_daily_snapshot(current_date)

        # ── 4. Check Correlated Drawdown Brake ──
        if self.config.enable_drawdown_brake:
            self._evaluate_drawdown_brake(current_date)

        # ── 5. Evening Scan & Setup Screening for Day T+1 ──
        self._run_evening_scan(current_date)

    def _evaluate_active_position_exits(self, current_date: date) -> None:
        """
        Checks active positions against the 5 precision exit rules using Day T's candle.
        """
        remaining_positions: List[BacktestPosition] = []

        for pos in self.active_positions:
            candle = self.data_loader.get_candle_for_date(pos.ticker_symbol, current_date)

            if candle is None:
                # Stock didn't trade today (e.g. trading halt or holiday).
                # Do not increment holding time when no session occurred.
                remaining_positions.append(pos)
                continue

            pos.days_held += 1

            exit_res = PrecisionExitEngine.evaluate_active_trade(
                entry_price=pos.entry_price,
                stop_loss=pos.initial_stop_loss,
                target_price=pos.target_price,
                invalidation_level=pos.invalidation_level,
                days_held=pos.days_held,
                daily_candle=candle,
                has_upcoming_earnings=False,
            )

            if exit_res.is_exit_triggered:
                # Position exited!
                exit_price = exit_res.exit_price
                gross_exit_val = pos.shares * exit_price
                gross_pnl = gross_exit_val - pos.capital_invested

                # Calculate statutory exit friction
                if self.config.enable_friction:
                    exit_fric_breakdown = PositionSizingEngine.calculate_statutory_friction(
                        shares=pos.shares,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                    )
                    # We already deducted entry friction when entering; exit friction is the remainder
                    # To be precise, calculate pure exit leg friction + DP charges
                    total_friction = exit_fric_breakdown.total_friction_inr
                    exit_friction = max(total_friction - pos.entry_friction_inr, 0.0)
                else:
                    total_friction = 0.0
                    exit_friction = 0.0

                net_exit_cash = gross_exit_val - exit_friction
                self.cash += net_exit_cash
                net_pnl = gross_pnl - total_friction
                net_ret_pct = (net_pnl / pos.capital_invested) * 100.0 if pos.capital_invested > 0 else 0.0

                closed_trade = BacktestTrade(
                    trade_id=pos.position_id,
                    ticker_symbol=pos.ticker_symbol,
                    sector=pos.sector,
                    setup_type=pos.setup_type,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    shares=pos.shares,
                    capital_invested=pos.capital_invested,
                    initial_stop_loss=pos.initial_stop_loss,
                    target_price=pos.target_price,
                    invalidation_level=pos.invalidation_level,
                    exit_date=current_date,
                    exit_price=exit_price,
                    exit_reason=exit_res.exit_reason,
                    holding_days=pos.days_held,
                    gross_pnl_inr=round(gross_pnl, 2),
                    entry_friction_inr=round(pos.entry_friction_inr, 2),
                    exit_friction_inr=round(exit_friction, 2),
                    total_friction_inr=round(total_friction, 2),
                    net_pnl_inr=round(net_pnl, 2),
                    net_return_pct=round(net_ret_pct, 2),
                    realized_r_multiple=exit_res.realized_r_multiple,
                )
                self.closed_trades.append(closed_trade)
            else:
                remaining_positions.append(pos)

        self.active_positions = remaining_positions

    def _execute_pending_orders(self, current_date: date) -> None:
        """
        Executes pending GTT limit entry orders generated from yesterday's scan.
        """
        if not self.pending_orders:
            return

        for order in self.pending_orders:
            # Check capacity limits before executing
            if len(self.active_positions) >= self.config.max_concurrent_positions:
                continue

            sector_count = sum(1 for p in self.active_positions if p.sector == order.sector)
            if sector_count >= self.config.max_per_sector:
                continue

            candle = self.data_loader.get_candle_for_date(order.ticker_symbol, current_date)
            if candle is None:
                continue

            # Realism: Check if price reached the entry level
            c_low = candle["low"]
            c_high = candle["high"]
            c_open = candle["open"]

            # Buy-limit execution:
            # If the market opens at/below the limit, fill at the actual open
            # (price improvement). Otherwise, fill at the limit only if traded down to it.
            is_filled = False
            fill_price = order.entry_price

            if c_open <= order.entry_price:
                is_filled = True
                fill_price = c_open
            elif c_low <= order.entry_price <= c_high:
                is_filled = True
                fill_price = order.entry_price

            if is_filled:
                capital_needed = order.shares * fill_price
                entry_friction = order.estimated_entry_friction_inr if self.config.enable_friction else 0.0

                total_cash_needed = capital_needed + entry_friction
                if self.cash >= total_cash_needed:
                    self.cash -= total_cash_needed
                    new_pos = BacktestPosition(
                        position_id=str(uuid.uuid4())[:8],
                        ticker_symbol=order.ticker_symbol,
                        sector=order.sector,
                        setup_type=order.setup_type,
                        entry_date=current_date,
                        entry_price=fill_price,
                        shares=order.shares,
                        capital_invested=capital_needed,
                        initial_stop_loss=order.stop_loss,
                        target_price=order.target_price,
                        invalidation_level=order.invalidation_level,
                        entry_friction_inr=entry_friction,
                        capital_at_risk_net_inr=order.capital_at_risk_net_inr,
                        days_held=0,
                    )
                    self.active_positions.append(new_pos)

        # Orders that did not fill today expire.
        self.pending_orders = []

    def _record_daily_snapshot(self, current_date: date) -> None:
        """
        Marks all active positions to market and records daily equity point.
        """
        invested_market_value = 0.0
        for pos in self.active_positions:
            candle = self.data_loader.get_candle_for_date(pos.ticker_symbol, current_date)
            price = candle["close"] if candle else pos.entry_price
            invested_market_value += pos.shares * price

        total_equity = self.cash + invested_market_value
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        dd_pct = ((self.peak_equity - total_equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        snapshot = DailyEquityPoint(
            trade_date=current_date,
            cash_inr=round(self.cash, 2),
            invested_capital_inr=round(invested_market_value, 2),
            total_equity_inr=round(total_equity, 2),
            open_positions_count=len(self.active_positions),
            drawdown_pct=round(dd_pct, 2),
        )
        self.daily_equity_curve.append(snapshot)

    def _evaluate_drawdown_brake(self, current_date: date) -> None:
        """
        Enforces correlated-drawdown circuit breaker:
        If >= 3 stop-outs occur in 7 rolling calendar days, pause new entries for 7 days.
        """
        window_start = current_date - timedelta(days=7)
        recent_stop_outs = [
            t
            for t in self.closed_trades
            if t.exit_date >= window_start
            and t.exit_reason in (ExitReason.STOP_LOSS_HIT.value, ExitReason.OVERNIGHT_GAP_STOP.value)
        ]

        if len(recent_stop_outs) >= 3:
            pause_end = current_date + timedelta(days=7)
            if self.system_paused_until is None or pause_end > self.system_paused_until:
                self.system_paused_until = pause_end
                logger.warning(
                    f"[{current_date}] Drawdown Circuit Breaker Triggered: "
                    f"{len(recent_stop_outs)} stop-outs in past 7 days. "
                    f"Pausing entries until {pause_end}."
                )

    def _run_evening_scan(self, current_date: date) -> None:
        """
        Executes institutional multi-gate screening on Day T close with strict
        zero look-ahead bias. Approved candidates become pending orders for Day T+1.
        """
        # 1. Check if paused by correlated-drawdown brake
        if self.system_paused_until and current_date < self.system_paused_until:
            return

        # 2. Check portfolio capacity
        available_slots = self.config.max_concurrent_positions - len(self.active_positions)
        if available_slots <= 0:
            return

        # 3. Get Point-in-Time Slice up to Day T
        nifty_slice, candidate_data = self.data_loader.get_point_in_time_slice(
            as_of_date=current_date,
            min_bars_required=self.config.min_warmup_bars,
        )

        # 4. Broad Market Regime Clearance
        if self.config.enable_regime_filter and not nifty_slice.empty:
            regime = MarketRegimeEngine.evaluate_nifty_regime(nifty_slice)
            if not regime.get("allow_swing_longs", True):
                return

        # 5. Expiry Week status
        is_expiry_week = self.expiry_filter.is_expiry_week_active(current_date).get("is_expiry_week", False)

        # 6. Screen candidates
        approved_orders: List[PendingOrder] = []
        active_tickers = {p.ticker_symbol for p in self.active_positions}

        for ticker, data in candidate_data.items():
            if ticker in active_tickers:
                continue

            sector = data.get("sector", "UNKNOWN")
            active_sector_count = sum(1 for p in self.active_positions if p.sector == sector)
            if active_sector_count >= self.config.max_per_sector:
                continue

            df_stock = data.get("ohlcv")
            delivery_history = data.get("delivery_history", [])
            today_delivery = data.get("today_delivery", 0)

            # Gate B: Scan 6 core swing setups
            signals = SwingSetupScanner.scan_all_setups(
                ticker_symbol=ticker,
                df=df_stock,
                delivery_history=delivery_history,
                is_expiry_week=is_expiry_week,
            )
            if not signals:
                continue

            top_signal = signals[0]

            # Gate C: Institutional Delivery Volume Expansion.
            # Missing delivery data must never silently bypass this gate.
            if not delivery_history or today_delivery is None:
                continue

            del_check = NSEArchiveDeliveryClient().calculate_delivery_expansion(
                recent_delivery_history=delivery_history,
                today_delivery=today_delivery,
                is_expiry_week=is_expiry_week,
            )
            if not del_check["is_institutional_accumulation"]:
                continue

            # Gate D: Precision Risk Coordinates (+2R Check)
            risk_per_share = top_signal.entry_price - top_signal.stop_loss
            if risk_per_share <= 0:
                continue

            rr_ratio = (top_signal.target_price - top_signal.entry_price) / risk_per_share
            if rr_ratio < self.config.min_reward_risk:
                continue

            # Gate E: Friction-Adjusted Position Sizing
            current_portfolio_equity = self.daily_equity_curve[-1].total_equity_inr if self.daily_equity_curve else self.cash
            sizing = PositionSizingEngine.calculate_position_size(
                ticker_symbol=ticker,
                entry_price=top_signal.entry_price,
                stop_loss=top_signal.stop_loss,
                target_price=top_signal.target_price,
                total_capital=current_portfolio_equity,
                max_risk_pct=self.config.max_risk_pct,
            )

            if not sizing.is_executable or sizing.shares_to_buy <= 0:
                continue

            # Gate F: Portfolio Circuit Breaker Validation
            if self.config.enable_circuit_breakers:
                active_pos_dicts = [
                    {
                        "ticker_symbol": p.ticker_symbol,
                        "sector": p.sector,
                        "capital_at_risk_net_inr": p.capital_at_risk_net_inr,
                    }
                    for p in self.active_positions
                ]
                cb = PortfolioCircuitBreakers.evaluate_candidate_capacity(
                    active_positions=active_pos_dicts,
                    candidate_ticker=ticker,
                    candidate_sector=sector,
                    candidate_risk_inr=sizing.capital_at_risk_net_inr,
                    total_capital=current_portfolio_equity,
                )
                if not cb.is_allowed:
                    continue

            # Gate G: Dual-Layer Mansfield Relative Strength
            if self.config.enable_rs_filter:
                sector_series = data.get("sector_series")
                nifty_series = data.get("nifty_series")
                stock_series = df_stock["close"] if "close" in df_stock.columns else df_stock["Close"]

                if sector_series is not None and nifty_series is not None and len(sector_series) >= 50 and len(nifty_series) >= 50:
                    rs_result = RelativeStrengthEngine.evaluate_dual_layer_relative_strength(
                        stock_close=stock_series,
                        sector_close=sector_series,
                        nifty_close=nifty_series,
                        stock_symbol=ticker,
                        sector_name=sector,
                    )
                    if not rs_result["passes_rs_filter"]:
                        continue

            # Approved! Create pending order
            pending = PendingOrder(
                ticker_symbol=ticker,
                sector=sector,
                setup_type=top_signal.setup_type,
                entry_price=top_signal.entry_price,
                stop_loss=top_signal.stop_loss,
                target_price=top_signal.target_price,
                invalidation_level=top_signal.invalidation_level,
                shares=sizing.shares_to_buy,
                capital_required_inr=sizing.capital_required_inr,
                capital_at_risk_net_inr=sizing.capital_at_risk_net_inr,
                estimated_entry_friction_inr=sizing.friction.total_friction_inr,
                risk_reward_ratio=round(rr_ratio, 2),
                scan_date=current_date,
            )
            approved_orders.append(pending)

        # Sort approved orders by risk_reward_ratio descending and pick top available slots
        approved_orders.sort(key=lambda o: o.risk_reward_ratio, reverse=True)
        self.pending_orders = approved_orders[:available_slots]

    def _close_all_open_positions(self, final_date: date, reason: str = "BACKTEST_END") -> None:
        """Closes remaining active positions on final simulation session."""
        for pos in self.active_positions:
            candle = self.data_loader.get_candle_for_date(pos.ticker_symbol, final_date)
            exit_price = candle["close"] if candle else pos.entry_price
            gross_val = pos.shares * exit_price
            gross_pnl = gross_val - pos.capital_invested

            friction_breakdown = PositionSizingEngine.calculate_statutory_friction(
                shares=pos.shares,
                entry_price=pos.entry_price,
                exit_price=exit_price,
            )
            total_friction = friction_breakdown.total_friction_inr if self.config.enable_friction else 0.0
            exit_friction = max(total_friction - pos.entry_friction_inr, 0.0)

            self.cash += gross_val - exit_friction
            net_pnl = gross_pnl - total_friction
            net_ret_pct = (net_pnl / pos.capital_invested) * 100.0 if pos.capital_invested > 0 else 0.0

            risk_per_share = pos.entry_price - pos.initial_stop_loss
            r_mult = round((exit_price - pos.entry_price) / risk_per_share, 2) if risk_per_share > 0 else 0.0

            closed = BacktestTrade(
                trade_id=pos.position_id,
                ticker_symbol=pos.ticker_symbol,
                sector=pos.sector,
                setup_type=pos.setup_type,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                shares=pos.shares,
                capital_invested=pos.capital_invested,
                initial_stop_loss=pos.initial_stop_loss,
                target_price=pos.target_price,
                invalidation_level=pos.invalidation_level,
                exit_date=final_date,
                exit_price=exit_price,
                exit_reason=reason,
                holding_days=pos.days_held,
                gross_pnl_inr=round(gross_pnl, 2),
                entry_friction_inr=round(pos.entry_friction_inr, 2),
                exit_friction_inr=round(exit_friction, 2),
                total_friction_inr=round(total_friction, 2),
                net_pnl_inr=round(net_pnl, 2),
                net_return_pct=round(net_ret_pct, 2),
                realized_r_multiple=r_mult,
            )
            self.closed_trades.append(closed)

        self.active_positions = []

        # Reconcile the final equity-curve point with post-liquidation cash.
        # The normal daily snapshot is recorded before forced liquidation.
        final_equity = self.cash
        if final_equity > self.peak_equity:
            self.peak_equity = final_equity
        dd_pct = ((self.peak_equity - final_equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0
        final_snapshot = DailyEquityPoint(
            trade_date=final_date,
            cash_inr=round(self.cash, 2),
            invested_capital_inr=0.0,
            total_equity_inr=round(final_equity, 2),
            open_positions_count=0,
            drawdown_pct=round(dd_pct, 2),
        )
        if self.daily_equity_curve and self.daily_equity_curve[-1].trade_date == final_date:
            self.daily_equity_curve[-1] = final_snapshot
        else:
            self.daily_equity_curve.append(final_snapshot)
