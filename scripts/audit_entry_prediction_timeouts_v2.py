"""Audit unresolved entry-prediction outcomes using dataset MFE/MAE only."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
DEFAULT_DATASET = Path("data/prediction/entry_prediction_dataset_v1.csv")
REQUIRED = ["trade_date","ticker_symbol","setup_type","entry_price","risk_points","outcome_valid","label_target_before_stop","mfe_r","mae_r"]
def audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    unresolved = df[df["outcome_valid"] == 0].copy()
    print("ENTRY_PREDICTION_V1_TIMEOUT_AUDIT")
    print(f"rows={len(df)}")
    print(f"unresolved_rows={len(unresolved)} unresolved_rate={len(unresolved)/len(df):.4f}")
    if unresolved.empty:
        return df
    print("\nUNRESOLVED_MFE_MAE")
    print(unresolved[["mfe_r","mae_r"]].describe().to_string())
    mfe, mae = unresolved["mfe_r"], unresolved["mae_r"]
    print("\nUNRESOLVED_BUCKETS")
    for name, mask in [("mfe_ge_1r",mfe>=1),("mfe_ge_2r",mfe>=2),("mae_le_minus_1r",mae<=-1),("mae_le_minus_2r",mae<=-2)]:
        print(f"{name}={int(mask.sum())} rate={mask.mean():.4f}")
    print("\nUNRESOLVED_BY_SETUP")
    print(unresolved.groupby("setup_type").agg(rows=("setup_type","size"),mean_mfe_r=("mfe_r","mean"),median_mfe_r=("mfe_r","median"),mean_mae_r=("mae_r","mean"),median_mae_r=("mae_r","median")).to_string())
    print("\nUNRESOLVED_INTERPRETATION")
    print("Unresolved rows are not automatically losses; choose an explicit horizon-end outcome policy before retraining.")
    return df
def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET); audit(p.parse_args().dataset)
if __name__ == "__main__": main()
