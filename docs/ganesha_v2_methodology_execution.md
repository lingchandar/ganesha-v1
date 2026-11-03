# Ganesha V2 Methodology Execution

## Purpose
Build the updated methodology as a research-first, measurable shadow system before any production promotion.

## Lifecycle
DATA → VALIDATION → MARKET REGIME → SECTOR REGIME → STOCK STATE → SETUP → PREDICTION → CALIBRATION → RISK VALIDATION → DECISION → POSITION MONITORING → PROFIT MANAGEMENT → EXIT → OUTCOME → LEARNING

## Current baseline
- FYERS ingestion and daily-bar live methodology.
- TREND_PULLBACK validated live setup.
- Extra Trees candidate model is candidate-only.
- EntryGate and production thresholds are unchanged.
- Existing PrecisionExitEngine is reused for deterministic exit evaluation.

## Build order
1. Evaluation and outcome ledger schema.
2. Deterministic state machine in shadow mode.
3. Multi-level historical labels and OOS evaluators.
4. Thesis Strength 0–100 and ablation evaluator.
5. Earnings event schema and historical event dataset.
6. Gemini contextual adapter for earnings/news explanation; never a deterministic BUY/SELL controller.
7. Historical analogue engine.
8. Profit Release state machine extending existing exit/risk modules.
9. Full OOS comparison.
10. Production promotion only after evidence passes existing gates and stability review.

## Required measurable outputs
Every experiment must report trade count, coverage, win rate, average/median/expected R, profit factor, cumulative R, max drawdown in R, holding days, target/stop/time-exit rates, MFE/MAE, probability calibration (Brier/ECE), AUC/AP/PR-AUC when classification applies, top-k precision/lift, and regime/sector/ticker stability.

## Ablation ladder
Baseline setup → +candidate prediction → +state machine → +thesis strength → +earnings intelligence → +historical analogues → +profit release → full combined system.

The decision to keep a new component is based on untouched OOS improvement in expected R, profit factor, drawdown, calibration, coverage, and stability. A component that does not add OOS value is removed or retained only as diagnostic context.

## State machine
States: DISCOVERED, QUALIFYING, WATCH, READY, ENTRY_CONFIRMED, ACTIVE, PROFIT_MANAGEMENT, THESIS_WEAKENING, EXIT_PENDING, EXITED_PROFIT, EXITED_LOSS, TIME_EXPIRED, INVALIDATED, COOLDOWN.

All transitions use deterministic reason codes. The state machine does not place orders and does not bypass EntryGate.

## Prediction levels
- Setup validity.
- Favorable entry.
- T1/T2 reach.
- +1R/+2R/+3R before stop.
- Expected R.

Each probability must be trained and validated with time-safe labels; do not derive probabilities from future information during feature creation.

## Earnings intelligence
Track expected vs actual revenue/EPS/profit, growth, margins, guidance, cash flow, historical earnings pattern, valuation/expectations, and market/volume/sector reaction. Event states: APPROACHING, IMMINENT, RELEASED, BEAT, MISS, MIXED, POST_EARNINGS_CONFIRMATION, POST_EARNINGS_FAILURE.

## Historical analogues
For each historical event/setup, record whether +1R, +2R, +3R, or stop occurred within 1/3/5/10/15 sessions. Analogue selection must use only information known at the event time.

## Thesis Strength
Score 0–100 from setup quality, market/sector regime, price/volume confirmation, prediction evidence, earnings/catalyst context, and risk state. Start deterministic. Validate each component independently before assigning predictive weight.

## Profit Release
ENTRY → T1 → PARTIAL PROFIT → T2 → PARTIAL PROFIT → TRAILING → FINAL EXIT.
Possible actions: HOLD_FULL, HOLD, PARTIAL_EXIT, PROTECT_PROFIT, TRAIL, FULL_EXIT. Reuse the existing exit engine instead of duplicating stop/target/time logic.

## Team roles
- Data/Research: OHLCV, corporate events, sectors, data quality, leakage audit.
- Quant/Validation: labels, forward windows, R/MFE/MAE, walk-forward OOS, ablations.
- ML: multi-level prediction, calibration, model versioning.
- Earnings/LLM: event extraction and structured context; Gemini remains non-decisioning.
- Strategy/Decision: state machine, thesis strength, reason codes, allowed actions.
- Risk/Profit Management: sizing, R framework, partial exits, trailing, time/catalyst exits.
- Integration/QA: wiring, tests, schema checks, auditability, reproducibility.

## Hard constraints
No live order placement, no EntryGate bypass, no production threshold changes during this phase, no look-ahead leakage, no hard-coded earnings outcomes, and no promotion of candidate modules before untouched OOS validation.
