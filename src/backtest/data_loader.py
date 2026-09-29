"""
GANESHA V1 — Historical Backtest Data Loader

Efficiently loads candle history, sector classifications, and benchmark data
from PostgreSQL and local cache for bar-by-bar backtest simulation.
Guarantees strict Point-in-Time access with zero future data leakage.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text
from loguru import logger

from src.core.database import get_db_session


class HistoricalDataLoader:
    """
    Manages historical market data for all Nifty 100 constituent equities,
    sector benchmark series, and Nifty 50 index benchmark.
    """

    BENCHMARK_CACHE_PATH = Path("data/nifty50_benchmark.csv")

    def __init__(self):
        self.universe_df: pd.DataFrame = pd.DataFrame()
        self.stock_candles: Dict[str, pd.DataFrame] = {}
        self.ticker_sectors: Dict[str, str] = {}
        self.sector_series: Dict[str, pd.Series] = {}
        self.nifty_benchmark_df: pd.DataFrame = pd.DataFrame()
        self.trading_dates: List[date] = []
        self._is_loaded = False

    def load_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> None:
        """
        Loads universe, historical candles, and benchmark into memory.
        """
        logger.info("Loading backtest universe and candle data from PostgreSQL...")
        
        with get_db_session() as db:
            # 1. Load universe metadata
            u_query = text(
                """
                SELECT instrument_token, ticker_symbol, company_name, sector_name
                FROM nse_eligible_universe
                WHERE is_active_swing = TRUE
                ORDER BY ticker_symbol
                """
            )
            self.universe_df = pd.read_sql(u_query, db.connection())
            self.ticker_sectors = dict(
                zip(self.universe_df["ticker_symbol"], self.universe_df["sector_name"])
            )

            # 2. Load historical daily candles
            c_query = text(
                """
                SELECT 
                    u.ticker_symbol,
                    c.candle_timestamp,
                    c.open_price,
                    c.high_price,
                    c.low_price,
                    c.close_price,
                    c.volume_traded,
                    c.delivery_volume,
                    c.delivery_percentage
                FROM historical_daily_candles c
                JOIN nse_eligible_universe u ON c.instrument_token = u.instrument_token
                ORDER BY c.candle_timestamp ASC
                """
            )
            raw_candles_df = pd.read_sql(c_query, db.connection())

        if raw_candles_df.empty:
            raise RuntimeError("historical_daily_candles table is empty. Run candle_backfill first.")

        # Standardize timestamps to pure date
        raw_candles_df["trade_date"] = pd.to_datetime(raw_candles_df["candle_timestamp"]).dt.date

        # Apply date filters if specified
        if start_date:
            raw_candles_df = raw_candles_df[raw_candles_df["trade_date"] >= start_date]
        if end_date:
            raw_candles_df = raw_candles_df[raw_candles_df["trade_date"] <= end_date]

        # Organize by ticker
        self.stock_candles = {}
        for ticker, group in raw_candles_df.groupby("ticker_symbol"):
            df_t = group.sort_values("trade_date").copy()
            df_t.rename(
                columns={
                    "open_price": "open",
                    "high_price": "high",
                    "low_price": "low",
                    "close_price": "close",
                    "volume_traded": "volume",
                },
                inplace=True,
            )
            df_t.set_index("trade_date", inplace=True)
            self.stock_candles[ticker] = df_t

        # 3. Load or synthesize Nifty 50 benchmark
        self._load_nifty_benchmark(start_date, end_date)

        # 4. Synthesize Sector Composite Series
        self._compute_sector_series()

        # 5. Extract unique sorted trading dates
        all_dates = set()
        for df in self.stock_candles.values():
            all_dates.update(df.index)
        self.trading_dates = sorted(list(all_dates))

        self._is_loaded = True
        logger.success(
            f"Historical data ready: {len(self.stock_candles)} tickers, "
            f"{len(self.trading_dates)} trading sessions "
            f"({self.trading_dates[0]} to {self.trading_dates[-1]})."
        )

    def _load_nifty_benchmark(self, start_date: Optional[date], end_date: Optional[date]) -> None:
        """Loads Nifty 50 daily index data from cache or creates composite."""
        if self.BENCHMARK_CACHE_PATH.exists():
            df_bench = pd.read_csv(self.BENCHMARK_CACHE_PATH)
            date_col = "Date" if "Date" in df_bench.columns else df_bench.columns[0]
            df_bench["trade_date"] = pd.to_datetime(df_bench[date_col]).dt.date
            df_bench.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )
            df_bench.set_index("trade_date", inplace=True)
            if start_date:
                df_bench = df_bench[df_bench.index >= start_date]
            if end_date:
                df_bench = df_bench[df_bench.index <= end_date]
            self.nifty_benchmark_df = df_bench
        else:
            # Fallback composite benchmark: equal-weighted close of all stocks
            logger.warning("Nifty benchmark cache not found; generating synthetic composite benchmark.")
            close_matrix = pd.DataFrame(
                {sym: df["close"] for sym, df in self.stock_candles.items()}
            )
            composite_close = close_matrix.mean(axis=1)
            self.nifty_benchmark_df = pd.DataFrame(
                {
                    "open": composite_close,
                    "high": composite_close,
                    "low": composite_close,
                    "close": composite_close,
                    "volume": 1000000,
                },
                index=composite_close.index,
            )

    def _compute_sector_series(self) -> None:
        """
        Computes composite sector index series by averaging daily close prices
        of constituent stocks in each sector.
        """
        sector_stock_map: Dict[str, List[str]] = {}
        for ticker, sector in self.ticker_sectors.items():
            if ticker in self.stock_candles:
                sector_stock_map.setdefault(sector, []).append(ticker)

        self.sector_series = {}
        for sector, tickers in sector_stock_map.items():
            series_list = [self.stock_candles[t]["close"] for t in tickers if t in self.stock_candles]
            if series_list:
                sector_df = pd.concat(series_list, axis=1)
                self.sector_series[sector] = sector_df.mean(axis=1).sort_index()

    def get_point_in_time_slice(
        self,
        as_of_date: date,
        min_bars_required: int = 30,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Returns Nifty history and candidate data dictionary up to as_of_date.
        Strictly zero look-ahead bias: no row with trade_date > as_of_date is included.

        Returns:
            Tuple of (nifty_slice_df, candidate_data_dict)
        """
        if not self._is_loaded:
            raise RuntimeError("DataLoader must call load_data() before slicing.")

        # Benchmark slice
        nifty_slice = self.nifty_benchmark_df[self.nifty_benchmark_df.index <= as_of_date]

        candidate_data: Dict[str, Dict[str, Any]] = {}
        for ticker, df in self.stock_candles.items():
            df_slice = df[df.index <= as_of_date]
            if len(df_slice) < min_bars_required:
                continue

            sector = self.ticker_sectors.get(ticker, "UNKNOWN")
            sector_s = self.sector_series.get(sector)
            sector_slice = sector_s[sector_s.index <= as_of_date] if sector_s is not None else None
            nifty_series = nifty_slice["close"] if not nifty_slice.empty else None

            # Delivery must be point-in-time:
            # - history contains only sessions strictly before as_of_date
            # - today_delivery contains only the current session's value
            # Missing delivery data must remain missing; never synthesize a value.
            delivery_hist = []
            today_del = None
            if "delivery_volume" in df.columns:
                prior_del = df.loc[df.index < as_of_date, "delivery_volume"].dropna().tolist()
                delivery_hist = [
                    int(v)
                    for v in prior_del
                    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)
                ]

                current_del = df.loc[df.index == as_of_date, "delivery_volume"].dropna().tolist()
                if current_del:
                    value = current_del[-1]
                    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                        today_del = int(value)

            candidate_data[ticker] = {
                "ohlcv": df_slice.copy(),
                "delivery_history": delivery_history_to_int(delivery_hist),
                "today_delivery": today_del,
                "sector": sector,
                "sector_series": sector_slice,
                "nifty_series": nifty_series,
            }

        return nifty_slice, candidate_data

    def get_candle_for_date(self, ticker: str, trade_date: date) -> Optional[Dict[str, float]]:
        """
        Retrieves the exact daily candle for a stock on a specific date.
        """
        df = self.stock_candles.get(ticker)
        if df is None or trade_date not in df.index:
            return None
        row = df.loc[trade_date]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }


def delivery_history_to_int(raw_list: List[Any]) -> List[int]:
    """Helper to convert delivery volume entries to ints."""
    clean = []
    for item in raw_list:
        try:
            val = int(item)
            if val > 0:
                clean.append(val)
        except (ValueError, TypeError):
            continue
    return clean
