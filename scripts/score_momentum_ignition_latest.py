from src.prediction.momentum_ignition_scorer import score_latest


if __name__ == "__main__":
    scored = score_latest()
    columns = [
        "trade_date",
        "ticker_symbol",
        "ignition_probability",
        "return_1d",
        "return_3d",
        "volume_ratio_20d",
        "volume_zscore_20d",
        "range_atr_multiple",
        "atr_expansion_ratio",
        "relative_strength_20d",
        "sector_strength_20d",
    ]
    print(scored[columns].head(15).to_string(index=False))
