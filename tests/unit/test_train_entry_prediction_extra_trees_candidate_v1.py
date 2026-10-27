from pathlib import Path

import pandas as pd

from scripts.train_entry_prediction_extra_trees_candidate_v1 import train


def test_candidate_trainer_writes_candidate_bundle(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    output = tmp_path / "candidate.joblib"
    rows = []
    for day in range(240):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for ticker in ("AAA", "BBB"):
            rows.append({
                "trade_date": date,
                "rsi_14": 50.0,
                "rsi_change_5d": 0.1,
                "atr_pct": 0.02,
                "atr_expansion_ratio": 1.0,
                "volume_surge_multiple": 1.5,
                "risk_reward_ratio": 2.0,
                "distance_to_stop_atr": 1.0,
                "distance_to_target_atr": 2.0,
                "price_vs_ema20_pct": 0.01,
                "price_vs_ema50_pct": 0.02,
                "price_vs_ema200_pct": 0.03,
                "ema20_vs_ema50_pct": 0.01,
                "ema50_vs_ema200_pct": 0.01,
                "ema50_slope_20": 0.001,
                "return_5d": 0.02,
                "return_20d": 0.04,
                "volatility_20d": 0.02,
                "adx_14": 25.0,
                "plus_di_minus_di": 5.0,
                "macd_hist_pct": 0.001,
                "macd_hist_diff_pct": 0.0001,
                "setup_classic_resistance_breakout": 1.0 if ticker == "AAA" else 0.0,
                "setup_breakout_retest": 0.0,
                "setup_trend_pullback": 0.0,
                "setup_demand_support_bounce": 0.0,
                "setup_trend_continuation_flag": 0.0,
                "setup_rs_leader_breakout": 0.0,
                "label_target_before_stop": int((day + (0 if ticker == "AAA" else 1)) % 3 == 0),
            })
    pd.DataFrame(rows).to_csv(dataset, index=False)
    train(dataset, output)
    assert output.exists()
