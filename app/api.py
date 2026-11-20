"""SupportRouter FastAPI service (Phase 8): `POST /classify`, `GET /health`.

Thin layer over `supportrouter.predict`: the model is loaded once at startup (`lifespan`) from
the registry alias with its checksum verified; requests are validated with limits;
every prediction is logged as a keyed hash, never raw text. It never trains and never
calls an LLM. Run: `make serve`, then open `/docs`.
"""

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from supportrouter.config import Config, load_config
from supportrouter.predict import PredictionLog, Predictor

LOGGER = logging.getLogger("supportrouter.api")
LIMITS = load_config().api  # schema limits come from configs/default.yaml

Text = Annotated[str, StringConstraints(min_length=1, max_length=LIMITS.max_text_chars)]


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422

    texts: list[Text] = Field(min_length=1, max_length=LIMITS.max_batch)

    @field_validator("texts")
    @classmethod
    def no_blank_text(cls, texts: list[str]) -> list[str]:
        if any(not t.strip() for t in texts):
            raise ValueError("texts must not be empty or whitespace only")
        return texts


class ClassifyResult(BaseModel):
    intent: str
    category: str
    confidence: float
    needs_human_review: bool


class ClassifyResponse(BaseModel):
    request_id: str
    model_version: str
    latency_ms: float
    results: list[ClassifyResult]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    review_threshold: float
    log_hash: str


def create_app(cfg: Config | None = None) -> FastAPI:
    """App factory; tests pass a config that points at a tiny registered fixture model."""
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loaded once; a checksum mismatch raises here and the service refuses to start.
        app.state.predictor = Predictor.from_registry(cfg, cfg.api.model_alias)
        key = os.environ.get(cfg.api.log_key_env, "").encode() or None
        if key is None:
            LOGGER.warning("%s is not set: log uses plain SHA-256", cfg.api.log_key_env)
        raw = os.environ.get(cfg.api.raw_text_env, "0") == "1"
        app.state.log = PredictionLog(cfg.api.log_path, key, log_raw_text=raw)
        yield

    app = FastAPI(title="SupportRouter", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default body echoes the input; drop it so raw text never leaves.
        detail = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
        body = {"detail": detail, "request_id": request.state.request_id}
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.error("unhandled %s (request %s)", type(exc).__name__, request.state.request_id)
        body = {"detail": "internal error", "request_id": request.state.request_id}
        return JSONResponse(status_code=500, content=body)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> Any:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(status_code=503, content={"detail": "model not loaded"})
        return HealthResponse(
            status="ok",
            model_version=predictor.model_version,
            review_threshold=cfg.api.review_threshold,
            log_hash=request.app.state.log.hash_mode,
        )

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(body: ClassifyRequest, request: Request) -> Any:
        predictor: Predictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(status_code=503, content={"detail": "model not loaded"})
        start = time.perf_counter()
        preds = predictor.predict(body.texts)
        latency_ms = (time.perf_counter() - start) * 1000
        rid = request.state.request_id
        request.app.state.log.write(rid, body.texts, preds, latency_ms)
        threshold = cfg.api.review_threshold
        return ClassifyResponse(
            request_id=rid,
            model_version=predictor.model_version,
            latency_ms=latency_ms,
            results=[
                ClassifyResult(
                    intent=p.intent,
                    category=p.category,
                    confidence=p.confidence,
                    needs_human_review=p.confidence < threshold,
                )
                for p in preds
            ],
        )

    return app


app = create_app()
