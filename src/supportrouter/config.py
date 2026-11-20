"""Load `configs/default.yaml` into typed, immutable settings."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

# Set only inside Docker Compose so the API talks to the MLflow server service.
# A project-specific name, so a stray MLFLOW_TRACKING_URI in a shell cannot redirect runs.
TRACKING_URI_ENV = "SUPPORTROUTER_TRACKING_URI"


@dataclass(frozen=True)
class DataConfig:
    hf_dataset_id: str
    hf_revision: str
    raw_dir: Path
    raw_file: str

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.raw_file


@dataclass(frozen=True)
class SplitConfig:
    """Split ratios and the report-only near-duplicate threshold."""

    processed_dir: Path
    train: float
    val: float
    test: float
    near_duplicate_threshold: float


@dataclass(frozen=True)
class BaselineConfig:
    """B1 hyperparameters; `c_grid` is searched on validation only."""

    ngram_range: tuple[int, int]
    c_grid: tuple[float, ...]
    max_iter: int
    metric_tolerance: float


@dataclass(frozen=True)
class TransformerConfig:
    """T1 fine-tuning settings; the model is pinned to a commit like the dataset."""

    model_name: str
    revision: str
    max_length: int
    learning_rate: float
    epochs: int
    batch_size: int
    eval_batch_size: int
    warmup_ratio: float
    weight_decay: float
    import_val_tolerance: float


@dataclass(frozen=True)
class MlflowConfig:
    db_path: Path
    artifact_dir: Path
    experiment: str
    registered_model: str
    server_uri: str | None = None  # MLflow server URL; None = local SQLite store

    @property
    def tracking_uri(self) -> str:
        return self.server_uri or f"sqlite:///{self.db_path}"


@dataclass(frozen=True)
class EvaluationConfig:
    reports_dir: Path
    latency_calls: int
    latency_warmup: int
    predictions_dir: Path
    typo_rates: tuple[float, ...]
    typo_min_word_len: int


@dataclass(frozen=True)
class LLMConfig:
    """L1/L2 settings: provider, pinned model, retry bound, prices + cap."""

    provider: str
    model: str
    model_digest: str
    base_url: str
    timeout_s: float
    max_retries: int
    max_output_tokens: int
    price_in_per_1m: float
    price_out_per_1m: float
    cost_cap_usd: float
    subset_size: int
    few_shot_per_intent: int
    results_path: Path


@dataclass(frozen=True)
class ApiConfig:
    """Serving settings: alias, threshold, input limits and log."""

    model_alias: str
    review_threshold: float
    max_text_chars: int
    max_batch: int
    log_path: Path
    log_key_env: str
    raw_text_env: str


@dataclass(frozen=True)
class LabelConfig:
    """Known label sets; `intent_to_category` is read-only (MappingProxyType)."""

    intent_to_category: Mapping[str, str]

    @property
    def intents(self) -> tuple[str, ...]:
        return tuple(sorted(self.intent_to_category))

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.intent_to_category.values())))


@dataclass(frozen=True)
class QualityGateConfig:
    """CI quality gate: minimum B1 SAMPLE macro-F1 on validation."""

    min_val_macro_f1: float
    threshold_env: str
    report_path: Path


@dataclass(frozen=True)
class MonitoringConfig:
    """Traffic simulation and drift report (Phase 11)."""

    reference_share: float
    batch_size: int
    shift_categories: tuple[str, ...]
    shift_share: float
    shift_typo_rate: float
    batches_path: Path
    drift_share: float
    report_dir: Path


@dataclass(frozen=True)
class Config:
    seed: int
    sample_size: int
    data: DataConfig
    split: SplitConfig
    baselines: BaselineConfig
    transformer: TransformerConfig
    mlflow: MlflowConfig
    models_dir: Path
    evaluation: EvaluationConfig
    llm: LLMConfig
    api: ApiConfig
    quality_gate: QualityGateConfig
    monitoring: MonitoringConfig
    labels: LabelConfig


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ValueError(f"missing config key: {where}{key}")
    return section[key]


def _parse_labels(raw: Any) -> LabelConfig:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("config key 'labels' must be a non-empty mapping category -> intents")
    intent_to_category: dict[str, str] = {}
    for category, intents in raw.items():
        if not isinstance(intents, list) or not intents:
            raise ValueError(f"labels.{category} must be a non-empty list of intents")
        for intent in intents:
            if intent in intent_to_category:
                raise ValueError(f"intent {intent!r} is listed under more than one category")
            intent_to_category[str(intent)] = str(category)
    return LabelConfig(MappingProxyType(intent_to_category))


def _parse_split(raw: Any, root: Path) -> SplitConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'split' must be a mapping")
    ratios = {name: float(_require(raw, name, "split.")) for name in ("train", "val", "test")}
    if any(r <= 0 for r in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must be positive and sum to 1, got {ratios}")
    threshold = float(_require(raw, "near_duplicate_threshold", "split."))
    if not 0 < threshold <= 1:
        raise ValueError(f"split.near_duplicate_threshold must be in (0, 1], got {threshold}")
    return SplitConfig(
        processed_dir=root / str(_require(raw, "processed_dir", "split.")),
        near_duplicate_threshold=threshold,
        **ratios,
    )


def _parse_baselines(raw: Any) -> BaselineConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'baselines' must be a mapping")
    b1 = _require(raw, "b1", "baselines.")
    ngram = [int(n) for n in _require(b1, "ngram_range", "baselines.b1.")]
    if len(ngram) != 2 or not 1 <= ngram[0] <= ngram[1]:
        raise ValueError(f"baselines.b1.ngram_range must be [min, max], 1 <= min <= max: {ngram}")
    c_grid = tuple(float(c) for c in _require(b1, "c_grid", "baselines.b1."))
    if not c_grid or any(c <= 0 for c in c_grid):
        raise ValueError(f"baselines.b1.c_grid must be non-empty and positive: {c_grid}")
    return BaselineConfig(
        ngram_range=(ngram[0], ngram[1]),
        c_grid=c_grid,
        max_iter=int(_require(b1, "max_iter", "baselines.b1.")),
        metric_tolerance=float(_require(raw, "metric_tolerance", "baselines.")),
    )


def _parse_transformer(raw: Any) -> TransformerConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'transformer' must be a mapping")
    revision = str(_require(raw, "revision", "transformer."))
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"transformer.revision must be a 40-char commit SHA, got {revision!r}")
    tcfg = TransformerConfig(
        model_name=str(_require(raw, "model_name", "transformer.")),
        revision=revision,
        max_length=int(_require(raw, "max_length", "transformer.")),
        learning_rate=float(_require(raw, "learning_rate", "transformer.")),
        epochs=int(_require(raw, "epochs", "transformer.")),
        batch_size=int(_require(raw, "batch_size", "transformer.")),
        eval_batch_size=int(_require(raw, "eval_batch_size", "transformer.")),
        warmup_ratio=float(_require(raw, "warmup_ratio", "transformer.")),
        weight_decay=float(_require(raw, "weight_decay", "transformer.")),
        import_val_tolerance=float(_require(raw, "import_val_tolerance", "transformer.")),
    )
    positive = ("max_length", "learning_rate", "epochs", "batch_size", "eval_batch_size")
    if any(getattr(tcfg, name) <= 0 for name in positive):
        raise ValueError(f"transformer.{{{', '.join(positive)}}} must all be positive")
    if not 0 <= tcfg.warmup_ratio < 1:
        raise ValueError(f"transformer.warmup_ratio must be in [0, 1), got {tcfg.warmup_ratio}")
    return tcfg


def _parse_llm(raw: Any, root: Path) -> LLMConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'llm' must be a mapping")
    digest = str(_require(raw, "model_digest", "llm."))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"llm.model_digest must be a 64-char sha256 hex, got {digest!r}")
    lcfg = LLMConfig(
        provider=str(_require(raw, "provider", "llm.")),
        model=str(_require(raw, "model", "llm.")),
        model_digest=digest,
        base_url=str(_require(raw, "base_url", "llm.")),
        timeout_s=float(_require(raw, "timeout_s", "llm.")),
        max_retries=int(_require(raw, "max_retries", "llm.")),
        max_output_tokens=int(_require(raw, "max_output_tokens", "llm.")),
        price_in_per_1m=float(_require(raw, "price_in_per_1m", "llm.")),
        price_out_per_1m=float(_require(raw, "price_out_per_1m", "llm.")),
        cost_cap_usd=float(_require(raw, "cost_cap_usd", "llm.")),
        subset_size=int(_require(raw, "subset_size", "llm.")),
        few_shot_per_intent=int(_require(raw, "few_shot_per_intent", "llm.")),
        results_path=root / str(_require(raw, "results_path", "llm.")),
    )
    non_negative = ("max_retries", "price_in_per_1m", "price_out_per_1m", "cost_cap_usd")
    if any(getattr(lcfg, name) < 0 for name in non_negative):
        raise ValueError(f"llm.{{{', '.join(non_negative)}}} must not be negative")
    positive = ("timeout_s", "max_output_tokens", "subset_size", "few_shot_per_intent")
    if any(getattr(lcfg, name) <= 0 for name in positive):
        raise ValueError(f"llm.{{{', '.join(positive)}}} must all be positive")
    return lcfg


def _parse_api(raw: Any, root: Path) -> ApiConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'api' must be a mapping")
    acfg = ApiConfig(
        model_alias=str(_require(raw, "model_alias", "api.")),
        review_threshold=float(_require(raw, "review_threshold", "api.")),
        max_text_chars=int(_require(raw, "max_text_chars", "api.")),
        max_batch=int(_require(raw, "max_batch", "api.")),
        log_path=root / str(_require(raw, "log_path", "api.")),
        log_key_env=str(_require(raw, "log_key_env", "api.")),
        raw_text_env=str(_require(raw, "raw_text_env", "api.")),
    )
    if not 0 < acfg.review_threshold <= 1:
        raise ValueError(f"api.review_threshold must be in (0, 1], got {acfg.review_threshold}")
    if acfg.max_text_chars <= 0 or acfg.max_batch <= 0:
        raise ValueError("api.max_text_chars and api.max_batch must be positive")
    return acfg


def _parse_quality_gate(raw: Any, root: Path) -> QualityGateConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'quality_gate' must be a mapping")
    gate = QualityGateConfig(
        min_val_macro_f1=float(_require(raw, "min_val_macro_f1", "quality_gate.")),
        threshold_env=str(_require(raw, "threshold_env", "quality_gate.")),
        report_path=root / str(_require(raw, "report_path", "quality_gate.")),
    )
    if not 0 < gate.min_val_macro_f1 <= 1:
        raise ValueError(
            f"quality_gate.min_val_macro_f1 must be in (0, 1], got {gate.min_val_macro_f1}"
        )
    return gate


def _parse_monitoring(raw: Any, root: Path) -> MonitoringConfig:
    if not isinstance(raw, dict):
        raise ValueError("config key 'monitoring' must be a mapping")
    mon = MonitoringConfig(
        reference_share=float(_require(raw, "reference_share", "monitoring.")),
        batch_size=int(_require(raw, "batch_size", "monitoring.")),
        shift_categories=tuple(str(c) for c in _require(raw, "shift_categories", "monitoring.")),
        shift_share=float(_require(raw, "shift_share", "monitoring.")),
        shift_typo_rate=float(_require(raw, "shift_typo_rate", "monitoring.")),
        batches_path=root / str(_require(raw, "batches_path", "monitoring.")),
        drift_share=float(_require(raw, "drift_share", "monitoring.")),
        report_dir=root / str(_require(raw, "report_dir", "monitoring.")),
    )
    if not (0 < mon.reference_share < 1 and 0 < mon.shift_share < 1 and 0 < mon.drift_share <= 1):
        raise ValueError(
            "monitoring: reference_share and shift_share in (0, 1), drift_share in (0, 1]"
        )
    if not 0 <= mon.shift_typo_rate <= 1 or mon.batch_size <= 0 or not mon.shift_categories:
        raise ValueError("monitoring: shift_typo_rate in [0, 1], batch_size > 0, categories set")
    return mon


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read the YAML file, check required keys and return a frozen `Config`.

    Relative paths are resolved against the repository root (the parent of `configs/`).
    """
    path = path.resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    data = _require(raw, "data", "")
    if not isinstance(data, dict):
        raise ValueError("config key 'data' must be a mapping")

    revision = str(_require(data, "hf_revision", "data."))
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"data.hf_revision must be a 40-char commit SHA, got {revision!r}")

    tracking = _require(raw, "mlflow", "")
    if not isinstance(tracking, dict):
        raise ValueError("config key 'mlflow' must be a mapping")
    evaluation = _require(raw, "evaluation", "")
    if not isinstance(evaluation, dict):
        raise ValueError("config key 'evaluation' must be a mapping")
    root = path.parent.parent
    return Config(
        seed=int(_require(raw, "seed", "")),
        sample_size=int(_require(raw, "sample_size", "")),
        data=DataConfig(
            hf_dataset_id=str(_require(data, "hf_dataset_id", "data.")),
            hf_revision=revision,
            raw_dir=root / str(_require(data, "raw_dir", "data.")),
            raw_file=str(_require(data, "raw_file", "data.")),
        ),
        split=_parse_split(_require(raw, "split", ""), root),
        baselines=_parse_baselines(_require(raw, "baselines", "")),
        transformer=_parse_transformer(_require(raw, "transformer", "")),
        mlflow=MlflowConfig(
            db_path=root / str(_require(tracking, "db_path", "mlflow.")),
            artifact_dir=root / str(_require(tracking, "artifact_dir", "mlflow.")),
            experiment=str(_require(tracking, "experiment", "mlflow.")),
            registered_model=str(_require(tracking, "registered_model", "mlflow.")),
            server_uri=os.environ.get(TRACKING_URI_ENV) or None,
        ),
        models_dir=root / str(_require(raw, "models_dir", "")),
        evaluation=EvaluationConfig(
            reports_dir=root / str(_require(evaluation, "reports_dir", "evaluation.")),
            latency_calls=int(_require(evaluation, "latency_calls", "evaluation.")),
            latency_warmup=int(_require(evaluation, "latency_warmup", "evaluation.")),
            predictions_dir=root / str(_require(evaluation, "predictions_dir", "evaluation.")),
            typo_rates=tuple(float(r) for r in _require(evaluation, "typo_rates", "evaluation.")),
            typo_min_word_len=int(_require(evaluation, "typo_min_word_len", "evaluation.")),
        ),
        llm=_parse_llm(_require(raw, "llm", ""), root),
        api=_parse_api(_require(raw, "api", ""), root),
        quality_gate=_parse_quality_gate(_require(raw, "quality_gate", ""), root),
        monitoring=_parse_monitoring(_require(raw, "monitoring", ""), root),
        labels=_parse_labels(_require(raw, "labels", "")),
    )
