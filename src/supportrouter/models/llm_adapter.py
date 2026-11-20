"""Provider-agnostic LLM adapter for L1/L2.

Everything provider-specific lives in this file. `llm_zero_shot.py` only sees `LLMClient`.
The only real provider is a local Ollama model ($0); `MockLLM` exists for tests only and
never produces a quality number.
LLM output is untrusted text: it becomes a label only through `parse_intent`.
"""

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from supportrouter.config import LLMConfig

InvalidReason = Literal["not_json", "bad_schema", "unknown_label", "provider_error"]
MockMode = Literal["valid", "not_json", "bad_schema", "unknown_label", "mixed"]


@dataclass(frozen=True)
class LLMResponse:
    text: str  # raw model output, untrusted
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int
    ) -> LLMResponse: ...


class LLMProviderError(RuntimeError):
    """Network/HTTP/envelope failure; the caller may retry it (bounded)."""


class CostCapError(RuntimeError):
    """The estimated cost of a run is above the cost cap."""


@dataclass(frozen=True)
class ParsedOutput:
    intent: str | None
    valid: bool
    invalid_reason: InvalidReason | None


def parse_intent(text: str, labels: Sequence[str]) -> ParsedOutput:
    """Strict parser: exactly `{"intent": <known label>}` or invalid with a reason.

    No repair, no case folding, no closest-label guess: a wrong answer stays visible.
    """
    try:
        obj = json.loads(text)
    except ValueError:  # json.JSONDecodeError is a ValueError subclass
        return ParsedOutput(None, False, "not_json")
    if not isinstance(obj, dict) or set(obj) != {"intent"} or not isinstance(obj["intent"], str):
        return ParsedOutput(None, False, "bad_schema")
    if obj["intent"] not in labels:
        return ParsedOutput(None, False, "unknown_label")
    return ParsedOutput(obj["intent"], True, None)


def intent_json_schema(labels: Sequence[str]) -> dict[str, Any]:
    """JSON schema with the intent as an enum; Ollama constrains decoding to it."""
    return {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": list(labels)}},
        "required": ["intent"],
        "additionalProperties": False,
    }


def estimate_cost(
    n_messages: int,
    avg_in_tokens: float,
    avg_out_tokens: float,
    price_in_per_1m: float,
    price_out_per_1m: float,
) -> float:
    """USD = messages x (in_tokens x price_in + out_tokens x price_out) / 1,000,000."""
    per_message = avg_in_tokens * price_in_per_1m + avg_out_tokens * price_out_per_1m
    return n_messages * per_message / 1_000_000


def check_cost_cap(estimate_usd: float, cap_usd: float) -> None:
    """Refuse the run before the first call when the estimate is above the cap."""
    if estimate_usd > cap_usd:
        raise CostCapError(
            f"estimated cost ${estimate_usd:.4f} is above the cost cap ${cap_usd:.4f}"
        )


class MockLLM:
    """Deterministic fake for pipeline tests: output depends only on `user` and `mode`."""

    name = "mock"
    INPUT_TOKENS = 100
    OUTPUT_TOKENS = 10

    def __init__(self, labels: Sequence[str], mode: MockMode = "valid") -> None:
        self.labels = tuple(labels)
        self.mode = mode

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int
    ) -> LLMResponse:
        h = int(hashlib.sha256(user.encode("utf-8")).hexdigest(), 16)
        mode = self.mode
        if mode == "mixed":
            mode = ("valid", "not_json", "bad_schema", "unknown_label")[h % 4]
        texts = {
            "valid": json.dumps({"intent": self.labels[h % len(self.labels)]}),
            "not_json": "Sure! The intent is cancel_order.",
            "bad_schema": json.dumps({"intent": self.labels[0], "confidence": 0.9}),
            "unknown_label": json.dumps({"intent": "cancel_my_order"}),
        }
        return LLMResponse(texts[mode], self.INPUT_TOKENS, self.OUTPUT_TOKENS, 0.0)


class OllamaClient:
    """Local Ollama `/api/chat` over the standard library (no extra dependency)."""

    name = "ollama"

    def __init__(self, cfg: LLMConfig, labels: Sequence[str], seed: int) -> None:
        if urlparse(cfg.base_url).scheme not in {"http", "https"}:
            raise ValueError(f"llm.base_url must be http(s), got {cfg.base_url!r}")
        self.cfg = cfg
        self.seed = seed
        self.schema = intent_json_schema(labels)

    def _request(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise LLMProviderError(f"ollama {path} failed: {exc}") from exc
        if not isinstance(body, dict):
            raise LLMProviderError(f"ollama {path} returned a non-object body")
        return body

    def verify_model(self) -> None:
        """The local model blob must be the digest pinned in config."""
        models = self._request("/api/tags", None).get("models", [])
        digests = {m.get("name"): m.get("digest") for m in models if isinstance(m, dict)}
        if digests.get(self.cfg.model) != self.cfg.model_digest:
            raise ValueError(
                f"ollama model {self.cfg.model!r} has digest {digests.get(self.cfg.model)!r}, "
                f"config pins {self.cfg.model_digest!r}"
            )

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int
    ) -> LLMResponse:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": self.schema,
            "options": {"temperature": temperature, "seed": self.seed, "num_predict": max_tokens},
        }
        start = time.perf_counter()
        body = self._request("/api/chat", payload)
        latency_ms = (time.perf_counter() - start) * 1000
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMProviderError("ollama /api/chat returned no message content")
        return LLMResponse(
            text=message["content"],
            input_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
            latency_ms=latency_ms,
        )


def make_client(cfg: LLMConfig, labels: Sequence[str], seed: int) -> LLMClient:
    """Factory by config name; a real provider is checked against its pin before use."""
    if cfg.provider == "mock":
        return MockLLM(labels)
    if cfg.provider == "ollama":
        client = OllamaClient(cfg, labels, seed)
        client.verify_model()
        return client
    raise ValueError(f"unknown llm.provider {cfg.provider!r} (supported: ollama, mock)")
