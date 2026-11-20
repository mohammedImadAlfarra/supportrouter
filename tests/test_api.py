"""API tests: a tiny B1 fixture model as @champion; no real model, network or .env."""

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.api import LIMITS, create_app
from fastapi.testclient import TestClient

from conftest import make_cfg
from supportrouter import predict as predict_module
from supportrouter.config import Config
from supportrouter.predict import ModelChecksumError, text_hash
from supportrouter.train import register_served_models, train_baseline

SECRET_TEXT = "refund money back status 4242"


@pytest.fixture(scope="module")
def api_cfg(tmp_path_factory: pytest.TempPathFactory, mlflow_template: Path) -> Config:
    cfg = make_cfg(tmp_path_factory.mktemp("api"), mlflow_template)
    train_baseline(cfg, "b1")
    register_served_models(cfg, {"champion": "b1"})
    return cfg


@pytest.fixture
def client(api_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(api_cfg.api.log_key_env, "test-key")
    monkeypatch.delenv(api_cfg.api.raw_text_env, raising=False)
    with TestClient(create_app(api_cfg), raise_server_exceptions=False) as c:
        yield c


def log_lines(cfg: Config) -> list[dict[str, object]]:
    return [json.loads(x) for x in cfg.api.log_path.read_text().splitlines()]


def test_classify_happy_path(client: TestClient) -> None:
    r = client.post("/classify", json={"texts": ["refund money back status", "cancel my order"]})
    assert r.status_code == 200
    body = r.json()
    assert [x["intent"] for x in body["results"]] == ["track_refund", "cancel_order"]
    assert body["results"][0]["category"] == "REFUND"
    assert "(b1," in body["model_version"] and body["latency_ms"] >= 0
    assert r.headers["X-Request-ID"] == body["request_id"]
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["log_hash"] == "hmac-sha256"


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": ["x" * (LIMITS.max_text_chars + 1)]},  # oversize text
        {"texts": ["cancel my order"], "debug": True},  # extra field
        {"texts": [""]},  # empty text
        {"texts": ["   \n"]},  # whitespace only
        {"texts": ["hi"] * (LIMITS.max_batch + 1)},  # oversize batch
        {"texts": []},  # empty batch
        {"text": "cancel my order"},  # wrong field name
    ],
)
def test_invalid_requests_get_422_without_echoing_input(
    client: TestClient, payload: dict[str, object]
) -> None:
    r = client.post("/classify", json=payload)
    assert r.status_code == 422
    assert all("input" not in err for err in r.json()["detail"])  # no echo of the text
    assert "xxxxxxxx" not in r.text and "cancel my order" not in r.text
    assert r.json()["request_id"]


def test_log_line_has_keyed_hash_and_no_raw_text(client: TestClient, api_cfg: Config) -> None:
    rid = client.post("/classify", json={"texts": [SECRET_TEXT]}).json()["request_id"]
    [line] = [x for x in log_lines(api_cfg) if x["request_id"] == rid]
    expected = {"timestamp", "request_id", "text_hash", "intent", "confidence"}
    assert expected | {"model_version", "latency_ms"} <= set(line)
    assert line["text_hash"] == text_hash(SECRET_TEXT, b"test-key")
    assert line["text_hash"] != text_hash(SECRET_TEXT, None)  # keyed, not plain SHA-256
    assert SECRET_TEXT not in api_cfg.api.log_path.read_text()
    assert "4242" not in api_cfg.api.log_path.read_text()


def test_log_line_has_length_descriptors_only(client: TestClient, api_cfg: Config) -> None:
    """Text drift features are integers derived from the text, never the text itself."""
    rid = client.post("/classify", json={"texts": [SECRET_TEXT]}).json()["request_id"]
    [line] = [x for x in log_lines(api_cfg) if x["request_id"] == rid]
    assert line["text_chars"] == len(SECRET_TEXT)
    assert line["text_words"] == len(SECRET_TEXT.split())
    assert "text" not in line


@pytest.mark.parametrize(("threshold", "review"), [(1.0, True), (0.01, False)])
def test_needs_human_review_follows_the_threshold(
    api_cfg: Config, threshold: float, review: bool
) -> None:
    cfg = dataclasses.replace(
        api_cfg, api=dataclasses.replace(api_cfg.api, review_threshold=threshold)
    )
    with TestClient(create_app(cfg)) as c:
        result = c.post("/classify", json={"texts": ["cancel my order"]}).json()["results"][0]
    assert result["needs_human_review"] is review


def test_model_is_loaded_once_across_requests(
    api_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = predict_module.Predictor.from_registry.__func__  # type: ignore[attr-defined]

    def counting(cls: type, cfg: Config, alias: str = "champion") -> object:
        calls.append(alias)
        return original(cls, cfg, alias)

    monkeypatch.setattr(predict_module.Predictor, "from_registry", classmethod(counting))
    with TestClient(create_app(api_cfg)) as c:
        for _ in range(3):
            assert c.post("/classify", json={"texts": ["cancel my order"]}).status_code == 200
    assert calls == ["champion"]


def test_internal_error_is_generic_without_traceback(client: TestClient) -> None:
    def boom(texts: object) -> None:
        raise RuntimeError("secret detail")

    client.app.state.predictor.predict = boom  # type: ignore[attr-defined]
    r = client.post("/classify", json={"texts": ["cancel my order"]})
    assert r.status_code == 500
    assert r.json()["detail"] == "internal error"
    assert "secret detail" not in r.text and "Traceback" not in r.text


def test_tampered_model_stops_startup(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = make_cfg(tmp_path, mlflow_template)
    run = train_baseline(cfg, "b1")
    register_served_models(cfg, {"champion": "b1"})
    run.model_path.write_bytes(run.model_path.read_bytes() + b"x")
    with pytest.raises(ModelChecksumError), TestClient(create_app(cfg)):
        pass
