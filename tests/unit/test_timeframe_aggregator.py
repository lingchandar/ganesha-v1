import pandas as pd

from src.ingestion.timeframe_aggregator import TimeframeAggregator


def test_aggregates_complete_15_minute_bar():
    timestamps = pd.date_range("2026-09-29 09:15:00+00:00", periods=15, freq="min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open_price": range(100, 115),
        "high_price": range(101, 116),
        "low_price": range(99, 114),
        "close_price": range(100, 115),
        "volume_traded": [100] * 15,
    })

    result = TimeframeAggregator(15).aggregate(df)

    assert len(result) == 1
    assert result.iloc[0]["open_price"] == 100
    assert result.iloc[0]["high_price"] == 115
    assert result.iloc[0]["low_price"] == 99
    assert result.iloc[0]["close_price"] == 114
    assert result.iloc[0]["volume_traded"] == 1500
    assert result.iloc[0]["source_candle_count"] == 15


def test_drops_incomplete_higher_timeframe_bar():
    timestamps = pd.date_range("2026-09-29 09:15:00+00:00", periods=14, freq="min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open_price": [100] * 14,
        "high_price": [101] * 14,
        "low_price": [99] * 14,
        "close_price": [100] * 14,
        "volume_traded": [100] * 14,
    })

    assert TimeframeAggregator(15).aggregate(df).empty
