import pandas as pd

from scripts.evaluate_entry_prediction_strategy_v1 import LABEL_POLICY, _daily_top_k_rates, _dedupe_best_ticker


def test_dedupe_keeps_highest_probability_setup_per_ticker():
    frame = pd.DataFrame(
        {
            "ticker_symbol": ["A", "A", "B"],
            "probability": [0.60, 0.90, 0.70],
            "label_target_before_stop": [0, 1, 1],
        }
    )
    ranked = _dedupe_best_ticker(frame)
    assert ranked["ticker_symbol"].tolist() == ["A", "B"]
    assert ranked["probability"].tolist() == [0.90, 0.70]


def test_daily_top_k_rate_uses_one_candidate_per_ticker():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"] * 4),
            "ticker_symbol": ["A", "A", "B", "C"],
            "probability": [0.90, 0.80, 0.70, 0.60],
            "label_target_before_stop": [1, 0, 1, 0],
        }
    )
    rates = _daily_top_k_rates(frame, 2)
    assert rates == [1.0]


def test_oos_evaluator_uses_complete_horizon_label_policy():
    assert LABEL_POLICY == "target_before_stop_within_15_sessions_unresolved_as_loss"
