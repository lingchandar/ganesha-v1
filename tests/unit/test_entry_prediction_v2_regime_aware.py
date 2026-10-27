import pandas as pd

from scripts.evaluate_entry_prediction_v2_regime_aware import (
    _chronological_split,
    _matches,
    _parse_group,
)


def test_parse_group_round_trips_multiple_dimensions():
    group = "setup_type=TREND_PULLBACK | trend_regime=bullish | volatility_regime=high"
    assert _parse_group(group) == {
        "setup_type": "TREND_PULLBACK",
        "trend_regime": "bullish",
        "volatility_regime": "high",
    }


def test_matches_requires_all_group_dimensions():
    frame = pd.DataFrame(
        {
            "setup_type": ["TREND_PULLBACK", "TREND_PULLBACK", "BREAKOUT_RETEST"],
            "trend_regime": ["bullish", "bearish", "bullish"],
        }
    )
    mask = _matches(frame, {"setup_type": "TREND_PULLBACK", "trend_regime": "bullish"})
    assert mask.tolist() == [True, False, False]


def test_chronological_split_has_no_date_overlap():
    frame = pd.DataFrame({"trade_date": pd.date_range("2025-01-01", periods=10, freq="D")})
    development, holdout = _chronological_split(frame)
    assert development["trade_date"].max() < holdout["trade_date"].min()
    assert len(development) == 5
    assert len(holdout) == 5
