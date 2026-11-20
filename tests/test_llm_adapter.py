import dataclasses
import io
import json
import urllib.error
from typing import Any

import pytest

from supportrouter.config import load_config
from supportrouter.models import llm_adapter
from supportrouter.models.llm_adapter import (
    CostCapError,
    LLMProviderError,
    MockLLM,
    OllamaClient,
    check_cost_cap,
    estimate_cost,
    make_client,
    parse_intent,
)

CFG = load_config()
LABELS = CFG.labels.intents


def test_mock_invalid_json_is_counted_invalid() -> None:
    out = MockLLM(LABELS, "not_json").complete("sys", "cancel my order", max_tokens=32)
    parsed = parse_intent(out.text, LABELS)
    assert (parsed.valid, parsed.intent, parsed.invalid_reason) == (False, None, "not_json")


def test_mock_unknown_label_is_never_mapped_to_a_close_one() -> None:
    out = MockLLM(LABELS, "unknown_label").complete("sys", "cancel my order", max_tokens=32)
    assert json.loads(out.text)["intent"] == "cancel_my_order"  # one edit away from cancel_order
    parsed = parse_intent(out.text, LABELS)
    assert (parsed.valid, parsed.intent, parsed.invalid_reason) == (False, None, "unknown_label")


@pytest.mark.parametrize(
    "text",
    [
        '{"intent": "cancel_order", "confidence": 0.9}',  # extra key
        '{"label": "cancel_order"}',  # wrong key
        '["cancel_order"]',  # not an object
        '{"intent": 3}',  # not a string
    ],
)
def test_wrong_schema_is_counted_invalid(text: str) -> None:
    assert parse_intent(text, LABELS).invalid_reason == "bad_schema"


@pytest.mark.parametrize("text", ['{"intent": "Cancel_Order"}', '{"intent": " cancel_order"}'])
def test_no_case_or_space_repair(text: str) -> None:
    assert parse_intent(text, LABELS).invalid_reason == "unknown_label"


def test_valid_output_is_accepted() -> None:
    parsed = parse_intent('  {"intent": "cancel_order"}\n', LABELS)
    assert (parsed.valid, parsed.intent, parsed.invalid_reason) == (True, "cancel_order", None)


def test_mock_is_deterministic_and_mixed_covers_every_mode() -> None:
    mock = MockLLM(LABELS, "mixed")
    texts = [f"message {i}" for i in range(40)]
    first = [mock.complete("s", t, max_tokens=32).text for t in texts]
    assert first == [mock.complete("s", t, max_tokens=32).text for t in texts]
    reasons = {parse_intent(t, LABELS).invalid_reason for t in first}
    assert reasons == {None, "not_json", "bad_schema", "unknown_label"}


def test_estimate_cost_formula() -> None:
    # 300 messages x (400 in x $2 + 100 out x $10) / 1M = 300 x 1800 / 1M = $0.54
    assert estimate_cost(300, 400, 100, 2.0, 10.0) == pytest.approx(0.54)
    assert estimate_cost(300, 400, 100, CFG.llm.price_in_per_1m, CFG.llm.price_out_per_1m) == 0.0


def test_estimate_above_cap_aborts() -> None:
    estimate = estimate_cost(300, MockLLM.INPUT_TOKENS, MockLLM.OUTPUT_TOKENS, 2.0, 10.0)
    with pytest.raises(CostCapError, match="above the cost cap"):
        check_cost_cap(estimate, CFG.llm.cost_cap_usd)


def test_free_local_run_passes_the_zero_cap() -> None:
    assert CFG.llm.cost_cap_usd == 0.0
    check_cost_cap(estimate_cost(600, 1500, 10, 0.0, 0.0), CFG.llm.cost_cap_usd)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _fake_urlopen(bodies: dict[str, Any], seen: list[dict[str, Any]]) -> Any:
    def fake(req: Any, timeout: float) -> _FakeResponse:
        path = req.full_url.split("11434", 1)[1]
        if req.data is not None:
            seen.append(json.loads(req.data))
        return _FakeResponse(json.dumps(bodies[path]).encode())

    return fake


def test_ollama_request_is_temperature_0_seeded_and_schema_constrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []
    bodies = {
        "/api/tags": {"models": [{"name": CFG.llm.model, "digest": CFG.llm.model_digest}]},
        "/api/chat": {
            "message": {"content": '{"intent": "track_order"}'},
            "prompt_eval_count": 412,
            "eval_count": 9,
        },
    }
    monkeypatch.setattr(llm_adapter.urllib.request, "urlopen", _fake_urlopen(bodies, seen))
    client = make_client(CFG.llm, LABELS, seed=42)
    resp = client.complete("sys", "where is my package", max_tokens=32)
    assert (resp.text, resp.input_tokens, resp.output_tokens) == (
        '{"intent": "track_order"}',
        412,
        9,
    )
    options = seen[0]["options"]
    assert options == {"temperature": 0.0, "seed": 42, "num_predict": 32}
    assert seen[0]["format"]["properties"]["intent"]["enum"] == list(LABELS)


def test_ollama_model_with_other_digest_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies = {"/api/tags": {"models": [{"name": CFG.llm.model, "digest": "0" * 64}]}}
    monkeypatch.setattr(llm_adapter.urllib.request, "urlopen", _fake_urlopen(bodies, []))
    with pytest.raises(ValueError, match="config pins"):
        make_client(CFG.llm, LABELS, seed=42)


def test_ollama_network_failure_is_a_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(req: Any, timeout: float) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_adapter.urllib.request, "urlopen", down)
    with pytest.raises(LLMProviderError, match="connection refused"):
        OllamaClient(CFG.llm, LABELS, seed=42).complete("s", "u", max_tokens=32)


def test_non_http_base_url_is_refused() -> None:
    cfg = dataclasses.replace(CFG.llm, base_url="file:///etc/passwd")
    with pytest.raises(ValueError, match="http"):
        OllamaClient(cfg, LABELS, seed=42)
