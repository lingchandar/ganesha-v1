"""Print the exact versioned 50-stock Trend Pullback validation universe."""
from __future__ import annotations

from scripts.trend_pullback_validation_universe_v1 import TREND_PULLBACK_VALIDATION_SYMBOLS

print(f"TREND_PULLBACK_VALIDATION_UNIVERSE: {len(TREND_PULLBACK_VALIDATION_SYMBOLS)} stocks")
for index, symbol in enumerate(TREND_PULLBACK_VALIDATION_SYMBOLS, start=1):
    print(f"{index:02d}. {symbol}")
