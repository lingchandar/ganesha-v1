from datetime import date

import pandas as pd
import pytest

from src.backtest.data_loader import HistoricalDataLoader


def loader_with_membership(rows):
    loader = HistoricalDataLoader()
    loader.membership_history_df = pd.DataFrame(rows)
    return loader


def test_membership_includes_stock_on_effective_date():
    loader = loader_with_membership([
        {
            "universe_name": "NSE_SWING",
            "ticker_symbol": "NSE:AAA-EQ",
            "instrument_token": 1,
            "effective_from": date(2025, 1, 10),
            "effective_to": None,
        }
    ])

    assert loader._is_member_on_date("NSE:AAA-EQ", date(2025, 1, 10)) is True
    assert loader._is_member_on_date("NSE:AAA-EQ", date(2025, 1, 9)) is False


def test_membership_excludes_stock_on_effective_to_date():
    loader = loader_with_membership([
        {
            "universe_name": "NSE_SWING",
            "ticker_symbol": "NSE:AAA-EQ",
            "instrument_token": 1,
            "effective_from": date(2025, 1, 1),
            "effective_to": date(2025, 2, 1),
        }
    ])

    assert loader._is_member_on_date("NSE:AAA-EQ", date(2025, 1, 31)) is True
    assert loader._is_member_on_date("NSE:AAA-EQ", date(2025, 2, 1)) is False


def test_future_membership_does_not_leak_into_earlier_scan():
    loader = loader_with_membership([
        {
            "universe_name": "NSE_SWING",
            "ticker_symbol": "NSE:AAA-EQ",
            "instrument_token": 1,
            "effective_from": date(2026, 1, 1),
            "effective_to": None,
        }
    ])

    assert loader._is_member_on_date("NSE:AAA-EQ", date(2025, 12, 31)) is False


def test_missing_membership_history_fails_closed():
    loader = HistoricalDataLoader()

    with pytest.raises(RuntimeError, match="membership history is empty"):
        loader._is_member_on_date("NSE:AAA-EQ", date(2025, 1, 1))


def test_unknown_ticker_is_not_considered_a_member():
    loader = loader_with_membership([
        {
            "universe_name": "NSE_SWING",
            "ticker_symbol": "NSE:AAA-EQ",
            "instrument_token": 1,
            "effective_from": date(2025, 1, 1),
            "effective_to": None,
        }
    ])

    assert loader._is_member_on_date("NSE:BBB-EQ", date(2025, 1, 10)) is False



def test_build_swing_universe_requires_membership_and_liquidity():
    import pandas as pd
    from src.ingestion.liquidity_filter import LiquidityConfig
    from src.ingestion.universe_sync import build_swing_universe

    membership = pd.DataFrame(
        {
            "ticker_symbol": ["NSE:AAA-EQ", "NSE:BBB-EQ"],
            "instrument_token": [1, 2],
        }
    )
    good = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=60),
            "close_price": [100.0] * 60,
            "volume_traded": [200_000] * 60,
        }
    )

    result = build_swing_universe(
        membership,
        {"NSE:AAA-EQ": good},
        LiquidityConfig(),
    ).set_index("ticker_symbol")

    assert bool(result.loc["NSE:AAA-EQ", "eligible"]) is True
    assert bool(result.loc["NSE:BBB-EQ", "eligible"]) is False
    assert result.loc["NSE:BBB-EQ", "reason"] == "MISSING_LIQUIDITY_HISTORY"



def test_candle_backfill_uses_point_in_time_membership():
    from pathlib import Path

    source = Path("src/ingestion/candle_backfill.py").read_text()
    assert "FROM universe_membership_history" in source
    assert "effective_from <= :end_date" in source
    assert "(effective_to IS NULL OR effective_to > :end_date)" in source
    assert "FROM nse_eligible_universe" not in source
