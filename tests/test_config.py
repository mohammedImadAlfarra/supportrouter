from pathlib import Path

import pytest

from supportrouter.config import PROJECT_ROOT, TRACKING_URI_ENV, load_config

VALID_YAML = """
seed: 42
sample_size: 3000
data:
  hf_dataset_id: org/name
  hf_revision: 430d1a89bd93bd1fa23c16f29dd53e73f0087443
  raw_dir: data/raw
  raw_file: raw.csv
split:
  processed_dir: data/processed
  train: 0.7
  val: 0.15
  test: 0.15
  near_duplicate_threshold: 0.9
baselines:
  b1:
    ngram_range: [1, 2]
    c_grid: [1.0, 10.0]
    max_iter: 100
  metric_tolerance: 1.0e-9
transformer:
  model_name: distilbert-base-uncased
  revision: 12040accade4e8a0f71eabdb258fecc2e7e948be
  max_length: 64
  learning_rate: 5.0e-5
  epochs: 3
  batch_size: 16
  eval_batch_size: 64
  warmup_ratio: 0.1
  weight_decay: 0.01
  import_val_tolerance: 0.005
mlflow:
  db_path: mlruns/mlflow.db
  artifact_dir: mlruns/artifacts
  experiment: test
  registered_model: test-intent
models_dir: models
evaluation:
  reports_dir: reports/metrics
  latency_calls: 10
  latency_warmup: 2
  predictions_dir: reports/predictions
  typo_rates: [0.1, 0.2]
  typo_min_word_len: 4
llm:
  provider: mock
  model: "m:latest"
  model_digest: 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf
  base_url: "http://localhost:11434"
  timeout_s: 10
  max_retries: 2
  max_output_tokens: 32
  price_in_per_1m: 0.0
  price_out_per_1m: 0.0
  cost_cap_usd: 0.0
  subset_size: 300
  few_shot_per_intent: 2
  results_path: reports/llm.csv
api:
  model_alias: champion
  review_threshold: 0.98
  max_text_chars: 1000
  max_batch: 32
  log_path: logs/predictions.jsonl
  log_key_env: PREDICTION_LOG_KEY
  raw_text_env: LOG_RAW_TEXT
quality_gate:
  min_val_macro_f1: 0.85
  threshold_env: QUALITY_GATE_MIN_F1
  report_path: reports/quality_gate.json
monitoring:
  reference_share: 0.5
  batch_size: 300
  shift_categories: [ORDER]
  shift_share: 0.7
  shift_typo_rate: 0.2
  batches_path: logs/traffic_batches.json
  drift_share: 0.5
  report_dir: reports/drift
"""
LABELS_YAML = "labels:\n  ORDER:\n    - cancel_order\n"


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "configs" / "test.yaml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_uses_seed_42_and_pinned_revision() -> None:
    cfg = load_config()
    assert cfg.seed == 42
    assert len(cfg.data.hf_revision) == 40
    assert cfg.data.raw_path == PROJECT_ROOT / "data" / "raw" / "bitext_customer_support.csv"


def test_relative_paths_resolve_against_repo_root(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, VALID_YAML + LABELS_YAML))
    assert cfg.data.raw_path == tmp_path.resolve() / "data" / "raw" / "raw.csv"


def test_missing_key_raises(tmp_path: Path) -> None:
    text = (VALID_YAML + LABELS_YAML).replace("  raw_file: raw.csv\n", "")
    with pytest.raises(ValueError, match="missing config key: data.raw_file"):
        load_config(write_config(tmp_path, text))


def test_branch_name_revision_is_refused(tmp_path: Path) -> None:
    text = (VALID_YAML + LABELS_YAML).replace("430d1a89bd93bd1fa23c16f29dd53e73f0087443", "main")
    with pytest.raises(ValueError, match="40-char commit SHA"):
        load_config(write_config(tmp_path, text))


def test_intent_under_two_categories_is_refused(tmp_path: Path) -> None:
    text = VALID_YAML + "labels:\n  A:\n    - x\n  B:\n    - x\n"
    with pytest.raises(ValueError, match="more than one category"):
        load_config(write_config(tmp_path, text))


def test_default_split_is_70_15_15() -> None:
    split = load_config().split
    assert (split.train, split.val, split.test) == (0.70, 0.15, 0.15)
    assert split.processed_dir == PROJECT_ROOT / "data" / "processed"


@pytest.mark.parametrize("ratios", ["train: 0.8\n  val: 0.15", "train: 0.85\n  val: 0.0"])
def test_split_ratios_must_be_positive_and_sum_to_one(tmp_path: Path, ratios: str) -> None:
    text = (VALID_YAML + LABELS_YAML).replace("train: 0.7\n  val: 0.15", ratios)
    with pytest.raises(ValueError, match="positive and sum to 1"):
        load_config(write_config(tmp_path, text))


def test_default_b1_grid_and_ngrams() -> None:
    b = load_config().baselines
    assert b.ngram_range == (1, 2)
    assert b.c_grid == (0.1, 1.0, 10.0, 100.0, 1000.0)


def test_non_positive_c_is_refused(tmp_path: Path) -> None:
    text = (VALID_YAML + LABELS_YAML).replace("c_grid: [1.0, 10.0]", "c_grid: [0.0, 1.0]")
    with pytest.raises(ValueError, match="c_grid must be non-empty and positive"):
        load_config(write_config(tmp_path, text))


def test_default_transformer_is_pinned_with_max_length_64() -> None:
    t = load_config().transformer
    assert t.model_name == "distilbert-base-uncased"
    assert len(t.revision) == 40 and t.max_length == 64


def test_transformer_branch_revision_is_refused(tmp_path: Path) -> None:
    pinned = "revision: 12040accade4e8a0f71eabdb258fecc2e7e948be"
    text = VALID_YAML.replace(pinned, "revision: main")
    with pytest.raises(ValueError, match="transformer.revision"):
        load_config(write_config(tmp_path, text + LABELS_YAML))


@pytest.mark.parametrize("bad", ["warmup_ratio: 1.0", "epochs: 0"])
def test_invalid_transformer_values_are_refused(tmp_path: Path, bad: str) -> None:
    key = bad.split(":")[0]
    good = {"warmup_ratio": "warmup_ratio: 0.1", "epochs": "epochs: 3"}[key]
    with pytest.raises(ValueError, match="transformer"):
        load_config(write_config(tmp_path, VALID_YAML.replace(good, bad) + LABELS_YAML))


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("model_digest: 6577", "model_digest: latest6577", "64-char sha256"),
        ("max_retries: 2", "max_retries: -1", "must not be negative"),
        ("cost_cap_usd: 0.0", "cost_cap_usd: -5", "must not be negative"),
        ("subset_size: 300", "subset_size: 0", "must all be positive"),
    ],
)
def test_bad_llm_settings_are_refused(tmp_path: Path, old: str, new: str, match: str) -> None:
    path = write_config(tmp_path, VALID_YAML.replace(old, new) + LABELS_YAML)
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_tracking_uri_defaults_to_local_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    cfg = load_config()
    assert cfg.mlflow.server_uri is None
    assert cfg.mlflow.tracking_uri == f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"


def test_tracking_uri_env_points_at_mlflow_server(monkeypatch: pytest.MonkeyPatch) -> None:
    # Docker Compose sets this so the API reads the registry through the MLflow service.
    monkeypatch.setenv(TRACKING_URI_ENV, "http://mlflow:5000")
    assert load_config().mlflow.tracking_uri == "http://mlflow:5000"


def test_generic_mlflow_env_does_not_redirect_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://elsewhere:5000")
    assert load_config().mlflow.tracking_uri.startswith("sqlite:///")


@pytest.mark.parametrize(
    ("old", "new"),
    [("shift_share: 0.7", "shift_share: 1.0"), ("drift_share: 0.5", "drift_share: 0.0")],
)
def test_monitoring_shares_are_checked(tmp_path: Path, old: str, new: str) -> None:
    text = (VALID_YAML + LABELS_YAML).replace(old, new)
    with pytest.raises(ValueError, match="monitoring"):
        load_config(write_config(tmp_path, text))
