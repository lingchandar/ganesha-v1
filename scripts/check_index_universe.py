"""Print the current Nifty 50 + Nifty Bank controlled FYERS universe."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.nse_index_universe import fetch_index_constituents, load_controlled_index_universe

for name in ("NIFTY50", "BANKNIFTY"):
    symbols = fetch_index_constituents(name)
    print(f"{name}: {len(symbols)}")
    print("\n".join(symbols))

symbols = load_controlled_index_universe()
print(f"UNIQUE: {len(symbols)}")
