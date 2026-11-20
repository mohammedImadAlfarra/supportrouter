"""Inference only: load a hash-verified model once and predict.

No training imports. The served model is the registry alias `@champion`;
its local copy is loaded only if every file matches the SHA-256 logged with its run.
`evaluate.py` uses the same loader, so tests and serving share one code path.
"""

import hashlib
import hmac
import json
import logging
import pickle
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from supportrouter.config import Config
from supportrouter.data.load import file_sha256
from supportrouter.models import transformer as t1
from supportrouter.models.baseline import predict_with_scores

# texts -> (top intent, its probability); the same shape for every family (model-training 4).
PredictFn = Callable[[Sequence[str]], tuple[list[str], list[float]]]


class ModelChecksumError(RuntimeError):
    """The model file does not match the SHA-256 recorded for its MLflow run."""


@dataclass(frozen=True)
class LoadedModel:
    predict: PredictFn
    size_bytes: int  # bytes on disk of every model file


def load_checked_model(cfg: Config, run: mlflow.entities.Run) -> LoadedModel:
    """Load the local model only if its hashes equal the ones logged with the run."""
    model_dir = cfg.models_dir / str(run.info.run_name)
    if run.data.tags.get("family") == "t1":
        folder = model_dir / "model"
        logged = mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/local_copy/model.sha256")
        bad = t1.checksum_mismatches(folder, logged)
        if bad:
            raise ModelChecksumError(f"{bad} hash differs from run {run.info.run_id[:8]}")
        tokenizer, model = t1.load_t1(folder)
        max_length = cfg.transformer.max_length

        def predict_t1(texts: Sequence[str]) -> tuple[list[str], list[float]]:
            return t1.predict_with_scores(model, tokenizer, texts, max_length)

        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        return LoadedModel(predict_t1, size)

    path = model_dir / "model.pkl"
    logged = mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/local_copy/model.pkl.sha256")
    if file_sha256(path) != logged.split()[0]:
        raise ModelChecksumError(f"{path.name} hash differs from run {run.info.run_id[:8]}")
    sk_model = pickle.loads(path.read_bytes())  # noqa: S301 - hash-verified file we wrote

    def predict_sk(texts: Sequence[str]) -> tuple[list[str], list[float]]:
        return predict_with_scores(sk_model, texts)

    return LoadedModel(predict_sk, path.stat().st_size)


@dataclass(frozen=True)
class Prediction:
    intent: str
    category: str
    confidence: float  # top-class probability, compared with the review threshold
    model_version: str


class Predictor:
    """One loaded model; `predict` maps texts to intent, derived category and confidence."""

    def __init__(self, model: LoadedModel, intent_to_category: dict[str, str], version: str):
        self._model = model
        self._to_category = intent_to_category
        self.model_version = version

    @classmethod
    def from_registry(cls, cfg: Config, alias: str = "champion") -> "Predictor":
        """Resolve `<registered_model>@alias` to its run, then load the hash-checked local copy.

        Raises `ModelChecksumError` on any mismatch, so a service built on it refuses to start.
        """
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = mlflow.MlflowClient()
        name = cfg.mlflow.registered_model
        version = client.get_model_version_by_alias(name, alias)
        if version.run_id is None:
            raise LookupError(f"{name}@{alias} has no source run; cannot verify its checksum")
        run = client.get_run(version.run_id)
        model = load_checked_model(cfg, run)
        family = run.data.tags.get("family", "?")
        label = f"{name}/{version.version} ({family}, run {run.info.run_id[:8]})"
        return cls(model, dict(cfg.labels.intent_to_category), label)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        intents, scores = self._model.predict(texts)
        return [
            Prediction(i, self._to_category[i], float(s), self.model_version)
            for i, s in zip(intents, scores, strict=True)
        ]


LOGGER = logging.getLogger("supportrouter.predict")


def text_hash(text: str, key: bytes | None) -> str:
    """HMAC-SHA256 with the secret key, so short messages cannot be recovered by
    hashing guesses; plain SHA-256 only when no key is configured (startup warns)."""
    data = text.encode("utf-8")
    if key:
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    return hashlib.sha256(data).hexdigest()


class PredictionLog:
    """Append-only JSONL prediction log. Never stores raw text unless the
    explicit debug flag is on; a write failure is counted and reported without text."""

    def __init__(self, path: Path, key: bytes | None, log_raw_text: bool = False) -> None:
        self.path = path
        self.key = key
        self.log_raw_text = log_raw_text
        self.failures = 0
        self._lock = threading.Lock()  # sync endpoints run in a thread pool
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def hash_mode(self) -> str:
        return "hmac-sha256" if self.key else "sha256"

    def write(
        self, request_id: str, texts: Sequence[str], preds: Sequence[Prediction], latency_ms: float
    ) -> None:
        now = datetime.now(UTC).isoformat()
        lines = []
        for item, (text, pred) in enumerate(zip(texts, preds, strict=True)):
            record = {
                "timestamp": now,
                "request_id": request_id,
                "item": item,
                "text_hash": text_hash(text, self.key),
                "hash_mode": self.hash_mode,
                # Length descriptors for text drift; they cannot be reversed into the text.
                "text_chars": len(text),
                "text_words": len(text.split()),
                "intent": pred.intent,
                "confidence": round(pred.confidence, 6),
                "model_version": pred.model_version,
                "latency_ms": round(latency_ms, 3),
            }
            if self.log_raw_text:
                record["text"] = text
            lines.append(json.dumps(record))
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as exc:
            self.failures += 1
            LOGGER.error(
                "prediction log write failed (%s); failures=%d", type(exc).__name__, self.failures
            )
