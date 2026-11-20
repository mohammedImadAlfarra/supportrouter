import pytest

from supportrouter.config import Config
from supportrouter.predict import ModelChecksumError, Predictor
from supportrouter.train import register_served_models, train_baseline


def test_predictor_from_registry_returns_intent_category_confidence(cfg: Config) -> None:
    run = train_baseline(cfg, "b1")
    register_served_models(cfg, {"champion": "b1"})
    predictor = Predictor.from_registry(cfg)
    [pred] = predictor.predict(["refund money back status"])
    assert pred.intent == "track_refund" and pred.category == "REFUND"
    assert 0 < pred.confidence <= 1
    assert run.run_id[:8] in pred.model_version and "(b1," in pred.model_version


def test_tampered_model_refuses_to_load(cfg: Config) -> None:
    run = train_baseline(cfg, "b1")
    register_served_models(cfg, {"champion": "b1"})
    run.model_path.write_bytes(run.model_path.read_bytes() + b"x")
    with pytest.raises(ModelChecksumError, match="hash differs"):
        Predictor.from_registry(cfg)
