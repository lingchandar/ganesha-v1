"""
GANESHA V1 — Market Regime Engine & Bearish Circuit Breaker (Module 4)

REAL-TIME TRADING PHILOSOPHY:
"Knowing when NOT to trade is more valuable than forcing setups."
Swing momentum requires favorable broad market conditions.

Regime Definitions (Anchored to NIFTY 50 Index):
- BULLISH: Nifty Close > 50 EMA > 200 EMA and ADX(14) >= 18.
- NEUTRAL: Nifty consolidating between 50 EMA and 200 EMA.
- HIGH_VOLATILITY: Daily ATR% > 1.8% or India VIX > 22.0.
- BEARISH: Nifty Close < 50 EMA and Nifty Close < 200 EMA.

The Bearish Halt Circuit Breaker:
If Nifty 50 is classified as BEARISH, the engine immediately halts all
new swing long generation and explicitly outputs:
`STATUS: REGIME HALT — BEARISH MARKET CONDITIONS`
"""
from enum import Enum
from typing import Dict, Optional
import pandas as pd
from loguru import logger

from src.indicators.technical_indicators import TechnicalAnalysisEngine


class MarketRegimeState(str, Enum):
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    BEARISH = "BEARISH"


class MarketRegimeEngine:
    """
    Evaluates broader market regime from NIFTY 50 historical daily candles.
    """

    @classmethod
    def evaluate_nifty_regime(
        cls,
        nifty_daily_df: pd.DataFrame,
        india_vix: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Evaluate market regime using the trailing Nifty 50 daily candles.

        Args:
            nifty_daily_df: DataFrame containing at least 200 daily candles of Nifty 50
                            with [open_price, high_price, low_price, close_price, volume_traded].
            india_vix: Latest India VIX level (optional).

        Returns:
            Dict containing regime state, boolean trade permission, and diagnostic metrics.
        """
        if len(nifty_daily_df) < 200:
            logger.warning("Nifty daily history has fewer than 200 candles; using available data.")

        df = TechnicalAnalysisEngine.compute_all_features(nifty_daily_df)
        latest = df.iloc[-1]

        close = float(latest["close_price"])
        ema_50 = float(latest["ema_50"])
        ema_200 = float(latest["ema_200"])
        adx_14 = float(latest["adx_14"])
        atr_pct = float(latest["atr_pct"])

        # 1. High Volatility Check
        is_high_vol = (atr_pct > 1.8) or (india_vix is not None and india_vix > 22.0)

        # 2. Bearish Regime Check
        is_bearish = (close < ema_50) and (close < ema_200)

        # 3. Bullish Regime Check
        is_bullish = (close > ema_50 > ema_200) and (adx_14 >= 18.0)

        # Classify state
        if is_bearish:
            regime = MarketRegimeState.BEARISH
            allow_swing_longs = False
            status_message = "STATUS: REGIME HALT — BEARISH MARKET CONDITIONS"
        elif is_high_vol:
            regime = MarketRegimeState.HIGH_VOLATILITY
            allow_swing_longs = False
            status_message = "STATUS: REGIME HALT — HIGH VOLATILITY SPIKE DETECTED"
        elif is_bullish:
            regime = MarketRegimeState.BULLISH
            allow_swing_longs = True
            status_message = "STATUS: REGIME CLEAR — BULLISH SWING CONDITIONS"
        else:
            regime = MarketRegimeState.NEUTRAL
            allow_swing_longs = True  # Selective setups only (e.g. Demand Support Bounce)
            status_message = "STATUS: REGIME NEUTRAL — SELECTIVE CONVICTION ONLY"

        logger.info(
            f"Market Regime Evaluated: {regime.value} | Nifty Close: {close:.2f} | "
            f"50 EMA: {ema_50:.2f} | 200 EMA: {ema_200:.2f} | ADX: {adx_14:.2f} | ATR%: {atr_pct:.2f}%"
        )

        return {
            "regime": regime.value,
            "allow_swing_longs": allow_swing_longs,
            "status_message": status_message,
            "metrics": {
                "nifty_close": close,
                "nifty_ema_50": round(ema_50, 2),
                "nifty_ema_200": round(ema_200, 2),
                "nifty_adx_14": round(adx_14, 2),
                "nifty_atr_pct": round(atr_pct, 2),
                "india_vix": india_vix,
            }
        }
