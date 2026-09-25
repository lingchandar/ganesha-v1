"""
GANESHA V1 — Deterministic Technical Indicators & Market Structure Engine (Module 6)

SEBI & INSTITUTIONAL COMPLIANCE:
All indicator computations are purely deterministic mathematical functions.
Large Language Models NEVER calculate or invent indicator values.

Formulas Implemented:
- Exponential Moving Averages (EMA 20, 50, 200)
- Relative Strength Index (RSI 14) with Wilder's Smoothing
- Average True Range (ATR 14) and Volatility Expansion (ATR > SMA20(ATR))
- MACD (12, 26, 9) with Rising Histogram Detection (Delta Hist > 0)
- Average Directional Index (ADX 14) and Directional Movement (+DI, -DI)
- Linear Regression Trend Slope of 50 EMA (20-period window)
- Dynamic Support/Resistance Bands (Pivot +/- 0.75 * ATR)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class TechnicalAnalysisEngine:
    """
    Computes mathematical technical indicators on point-in-time daily OHLCV series.
    """

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average using standard smoothing alpha = 2 / (period + 1)."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR 14).
        TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
        ATR = Wilder's Exponential Smoothing of TR.
        """
        high = df["high_price"]
        low = df["low_price"]
        close = df["close_price"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder's RMA is equivalent to ewm(alpha=1/period, adjust=False)
        atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
        return atr

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI 14) using Wilder's smoothed averages.
        """
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Wilder's smoothing
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD Line, Signal Line, and Histogram.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX 14), +DI, and -DI using Wilder's smoothing.
        """
        high = df["high_price"]
        low = df["low_price"]
        close = df["close_price"]
        prev_close = close.shift(1)

        prev_high = high.shift(1)
        prev_low = low.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth series
        atr_smooth = pd.Series(true_range, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth)
        minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth)

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)

        return adx, plus_di, minus_di

    @staticmethod
    def calculate_ema_slope(ema_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate the 20-day linear regression slope of the 50 EMA.
        Positive slope indicates sustained trend momentum.
        """
        x = np.arange(window)
        x_mean = x.mean()
        x_diff = x - x_mean
        denom = (x_diff ** 2).sum()

        def get_slope(y):
            if len(y) < window or np.isnan(y).any():
                return 0.0
            y_diff = y - y.mean()
            return (x_diff * y_diff).sum() / denom

        return ema_series.rolling(window=window).apply(get_slope, raw=True)

    @classmethod
    def compute_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute full deterministic indicator suite for a stock's daily OHLCV history.
        DataFrame must contain: [open_price, high_price, low_price, close_price, volume_traded]
        """
        df = df.copy()
        close = df["close_price"]

        # EMAs
        df["ema_20"] = cls.calculate_ema(close, 20)
        df["ema_50"] = cls.calculate_ema(close, 50)
        df["ema_200"] = cls.calculate_ema(close, 200)

        # 50 EMA Trend Slope
        df["ema_50_slope_20"] = cls.calculate_ema_slope(df["ema_50"], 20)

        # RSI 14
        df["rsi_14"] = cls.calculate_rsi(close, 14)
        df["is_rsi_momentum_zone"] = (df["rsi_14"] >= 55.0) & (df["rsi_14"] <= 65.0)

        # ATR 14 & Volatility Expansion
        df["atr_14"] = cls.calculate_atr(df, 14)
        df["atr_pct"] = (df["atr_14"] / close) * 100.0
        df["atr_sma_20"] = df["atr_14"].rolling(window=20).mean()
        df["is_volatility_expanding"] = df["atr_14"] > df["atr_sma_20"]

        # MACD (12, 26, 9)
        macd, signal, hist = cls.calculate_macd(close, 12, 26, 9)
        df["macd_line"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        df["macd_hist_diff"] = hist.diff()
        df["is_macd_rising"] = (df["macd_hist"] > 0) & (df["macd_hist_diff"] > 0)

        # ADX 14
        adx, pdi, mdi = cls.calculate_adx(df, 14)
        df["adx_14"] = adx
        df["plus_di"] = pdi
        df["minus_di"] = mdi

        # Dynamic ATR Zones (+/- 0.75 * ATR around close)
        df["atr_dynamic_upper"] = close + (0.75 * df["atr_14"])
        df["atr_dynamic_lower"] = close - (0.75 * df["atr_14"])

        return df
