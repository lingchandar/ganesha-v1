import pandas as pd
import pytest

from src.prediction.momentum_ignition_scorer import MomentumIgnitionScorer


class DummyModel:
    def predict_proba(self, frame):
        return [[0.7, 0.3] for _ in range(len(frame))]


def test_scorer_adds_probability(monkeypatch):
    import src.prediction.momentum_ignition_scorer as module

    monkeypatch.setattr(
        module.joblib,
        "load",
        lambda _: {
            "model": DummyModel(),
            "features": ["return_1d", "volume_ratio_20d"],
            "label": "ignition_1d",
            "model_version": "momentum_ignition_v1",
        },
    )
    scorer = MomentumIgnitionScorer()
    frame = pd.DataFrame({"return_1d": [0.02], "volume_ratio_20d": [2.0]})
    result = scorer.score(frame)
    assert result.loc[0, "ignition_probability"] == pytest.approx(0.3)
    assert result.loc[0, "model_version"] == "momentum_ignition_v1"


def test_scorer_rejects_missing_features(monkeypatch):
    import src.prediction.momentum_ignition_scorer as module

    monkeypatch.setattr(
        module.joblib,
        "load",
        lambda _: {
            "model": DummyModel(),
            "features": ["return_1d", "volume_ratio_20d"],
            "label": "ignition_1d",
            "model_version": "momentum_ignition_v1",
        },
    )
    scorer = MomentumIgnitionScorer()
    with pytest.raises(ValueError, match="Missing scoring features"):
        scorer.score(pd.DataFrame({"return_1d": [0.02]}))
