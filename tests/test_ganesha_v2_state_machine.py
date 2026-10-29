from src.strategy.state_machine import GaneshaStateMachine, StateEvidence, TradeState


def test_shadow_lifecycle_transitions():
    machine = GaneshaStateMachine()
    assert machine.evaluate(StateEvidence(setup_valid=True)).current == TradeState.QUALIFYING
    assert machine.evaluate(StateEvidence(setup_valid=True, setup_ready=False)).current == TradeState.WATCH
    assert machine.evaluate(StateEvidence(setup_valid=True, setup_ready=True)).current == TradeState.READY
    assert machine.evaluate(StateEvidence(setup_valid=True, setup_ready=True, entry_confirmed=True)).current == TradeState.ENTRY_CONFIRMED
    assert machine.evaluate(StateEvidence(entry_executed=True)).current == TradeState.ACTIVE


def test_risk_exit_has_priority():
    machine = GaneshaStateMachine(TradeState.ACTIVE)
    transition = machine.evaluate(StateEvidence(exit_loss=True, thesis_score=90))
    assert transition.current == TradeState.EXITED_LOSS
    assert transition.reason_code == "EXIT_LOSS"


def test_thesis_weakening_is_shadow_only():
    machine = GaneshaStateMachine(TradeState.ACTIVE)
    transition = machine.evaluate(StateEvidence(thesis_score=49))
    assert transition.current == TradeState.THESIS_WEAKENING
