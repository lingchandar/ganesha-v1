import pandas as pd

from src.ingestion.liquidity_filter import LiquidityConfig, evaluate_liquidity


def make_candles(days=60, price=100.0, volume=200_000):
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=days, freq="D"),
            "close_price": [price] * days,
            "volume_traded": [volume] * days,
        }
    )


def test_liquidity_passes_for_consistently_traded_stock():
    result = evaluate_liquidity(make_candles())
    assert result["eligible"] is True
    assert result["reason"] == "LIQUIDITY_OK"


def test_liquidity_rejects_low_turnover():
    result = evaluate_liquidity(
        make_candles(volume=20_000),
        LiquidityConfig(min_average_daily_turnover=10_000_000),
    )
    assert result["eligible"] is False
    assert "average_turnover" in result["failed_checks"]


def test_liquidity_rejects_excess_zero_volume_days():
    candles = make_candles()
    candles.loc[:9, "volume_traded"] = 0
    result = evaluate_liquidity(candles)
    assert result["eligible"] is False
    assert "zero_volume" in result["failed_checks"]


def test_liquidity_fails_closed_when_history_is_short():
    result = evaluate_liquidity(
        make_candles(days=20),
        LiquidityConfig(min_observations=40),
    )
    assert result["eligible"] is False
    assert result["reason"] == "INSUFFICIENT_LIQUIDITY_HISTORY"
