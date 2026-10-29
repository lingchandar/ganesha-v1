from datetime import date

from src.earnings.schema import EarningsEvent, EarningsState
from src.earnings.context_adapter import DeterministicEarningsContextProvider
from src.strategy.historical_analogues import AnalogueEvent, summarize_analogues
from src.strategy.profit_release import ProfitAction, ProfitReleaseEngine, ProfitReleaseEvidence
from src.strategy.thesis_strength import ThesisEvidence, ThesisStrengthScorer
from src.validation.multi_level_labels import label_forward_window
from src.validation.outcome_ledger import OutcomeRecord


def test_thesis_strength_is_bounded_and_reasoned():
    result = ThesisStrengthScorer.score(ThesisEvidence(setup_quality=90, market_regime=80, sector_regime=80, price_volume_confirmation=90, prediction_probability=0.8, risk_quality=90))
    assert 0 <= result.score <= 100
    assert "THESIS_STRONG" in result.reason_codes


def test_profit_release_is_shadow_only_policy():
    result = ProfitReleaseEngine.decide(ProfitReleaseEvidence(current_r=2.2, thesis_score=80, momentum_score=75))
    assert result.action == ProfitAction.TRAIL
    assert result.reason_code == "STRONG_TREND_TRAILING"


def test_earnings_context_does_not_issue_trade_command():
    event = EarningsEvent("NSE:TCS-EQ", date(2026, 10, 1), EarningsState.RELEASED, expected_eps=10, actual_eps=11)
    result = DeterministicEarningsContextProvider().build_context("NSE:TCS-EQ", [event])
    assert "BUY" not in result.summary.upper()
    assert result.event == event


def test_historical_analogue_summary():
    events = [
        AnalogueEvent("a", "2026-01-01", (1.0, 2.0), 1, 1, 0, 0),
        AnalogueEvent("b", "2026-01-02", (1.1, 2.1), 1, 0, 0, 1),
    ]
    summary = summarize_analogues((1.05, 2.05), events, k=2)
    assert summary.count == 2
    assert 0 <= summary.one_r_probability <= 1


def test_multi_level_labels_are_forward_only():
    candles = __import__("pandas").DataFrame({"high": [105, 112, 120], "low": [99, 104, 110]})
    result = label_forward_window(candles, entry=100, stop=95, targets=[105, 110], max_days=3)
    assert result.setup_valid == 1
    assert result.t1_hit == 1


def test_outcome_record_serializes_dates():
    record = OutcomeRecord("NSE:TCS-EQ", "TREND_PULLBACK", date(2026, 10, 1), realized_r=1.5)
    payload = record.to_dict()
    assert payload["setup_date"] == "2026-10-01"
