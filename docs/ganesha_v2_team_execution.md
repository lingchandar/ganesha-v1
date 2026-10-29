# Ganesha V2 — Team Execution Plan

## Workstreams

1. **Data / Research**
   - Build point-in-time OHLCV, sector, corporate-event and earnings datasets.
   - Record data timestamp/source and reject future-known fields.
   - Output: reproducible event + market feature tables.

2. **Quant / Validation**
   - Build setup-valid, entry-favorable, T1/T2, +1R/+2R/+3R-before-stop labels.
   - Run chronological walk-forward OOS validation.
   - Report expected R, profit factor, drawdown, coverage, MFE/MAE, calibration, lift and stability.

3. **ML**
   - Train separate models per prediction level where sample size permits.
   - Version features/model bundles.
   - Keep candidate-only until untouched OOS gates pass.

4. **Earnings / LLM**
   - Convert earnings/news into structured facts and event states.
   - Gemini may summarize/extract context, but cannot emit deterministic BUY/SELL commands.
   - Measure whether earnings context improves OOS results over baseline.

5. **Strategy / Decision**
   - Integrate state machine, Thesis Strength, reason codes and allowed actions.
   - Never bypass EntryGate.
   - Shadow all new transitions before production.

6. **Risk / Profit Management**
   - Validate partial exits, protection and trailing in R terms.
   - Reuse PrecisionExitEngine and ActiveTradeManager.
   - Evaluate whether profit release improves expected R and drawdown OOS.

7. **Integration / QA**
   - Contract tests, schema validation, replay tests, leakage tests, reproducibility checks.
   - Verify no live order placement and no production threshold changes.

## Execution order

`Data → Labels → Baseline OOS → Multi-level ML → Thesis → Earnings → Analogues → Profit Release → Full Ablation → Shadow Integration → Production review`

No module is promoted because it looks good in-sample. Each module must demonstrate incremental untouched-OOS value and stability across time, regime, sector and ticker.
