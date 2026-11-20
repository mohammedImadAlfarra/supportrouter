"""T1 `train_transformer` + `evaluate_on_test` end to end on the tiny frozen split.

The "pretrained" model is a tiny random DistilBERT, so nothing is downloaded.
"""

import hashlib
import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from conftest import make_cfg
from supportrouter import train as train_module
from supportrouter.config import Config, TransformerConfig
from supportrouter.evaluate import evaluate_on_test
from supportrouter.predict import ModelChecksumError
from supportrouter.train import (
    EXPORT_RECORD,
    TrainResult,
    export_t1_run,
    import_t1_run,
    train_transformer,
)
from test_transformer import tiny_model, tiny_tokenizer


def fake_pretrained(tcfg: TransformerConfig, intents: object) -> tuple[object, object]:
    """No download: the 'pretrained' model is a 1-layer random DistilBERT."""
    return tiny_tokenizer(), tiny_model()


@pytest.fixture(autouse=True)
def tiny_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_module.t1, "load_pretrained", fake_pretrained)


@pytest.fixture(scope="module")
def t1_run(cfg: Config) -> TrainResult:
    with pytest.MonkeyPatch.context() as mp:  # module scope cannot use `monkeypatch`
        mp.setattr(train_module.t1, "load_pretrained", fake_pretrained)
        return train_transformer(cfg)


def test_t1_training_reads_only_train_and_val(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real = train_module.load_split

    def spy(c: Config, name: str) -> pd.DataFrame:
        seen.append(name)
        return real(c, name)

    monkeypatch.setattr(train_module, "load_split", spy)
    train_transformer(cfg, sample=30)
    assert seen == ["train", "val"]


def test_t1_sidecar_hashes_every_saved_file(t1_run: TrainResult) -> None:
    model_dir = t1_run.model_path
    lines = model_dir.with_name("model.sha256").read_text(encoding="utf-8").splitlines()
    listed = {name: digest for digest, name in (line.split("  ") for line in lines)}
    on_disk = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()}
    assert set(listed) == on_disk and "config.json" in on_disk
    for name, digest in listed.items():
        assert hashlib.sha256((model_dir / name).read_bytes()).hexdigest() == digest
    assert not (model_dir.parent / "trainer").exists()  # per-epoch checkpoints removed


def test_t1_run_logs_params_curves_and_split_hashes(cfg: Config, t1_run: TrainResult) -> None:
    run = mlflow.get_run(t1_run.run_id)
    assert run.data.tags["family"] == "t1" and run.data.tags["mode"] == "full"
    assert run.data.params["model_revision"] == cfg.transformer.revision
    assert run.data.params["split_sha256_test"]  # which frozen split the model belongs to
    curve = mlflow.MlflowClient().get_metric_history(t1_run.run_id, "epoch_train_loss")
    assert [m.step for m in curve] == list(range(1, cfg.transformer.epochs + 1))
    assert 1 <= int(run.data.params["best_epoch"]) <= cfg.transformer.epochs
    assert "val_intent_macro_f1" in run.data.metrics


def test_t1_artifacts_include_model_and_checksum(t1_run: TrainResult) -> None:
    names = {
        Path(a.path).name
        for folder in ("model", "local_copy")
        for a in mlflow.MlflowClient().list_artifacts(t1_run.run_id, folder)
    }
    assert {"config.json", "model.safetensors", "model.sha256"} <= names


def test_t1_test_report_written_with_t1_params(cfg: Config, t1_run: TrainResult) -> None:
    report = evaluate_on_test(cfg, "t1")
    saved = json.loads((cfg.evaluation.reports_dir / "t1.json").read_text())
    assert saved == report and report["family"] == "t1" and report["test_rows"] == 18
    assert report["params"]["best_epoch"] == str(t1_run.params["best_epoch"])  # MLflow: str
    assert report["model_size_bytes"] > 0
    assert report["latency"]["latency_p50_ms"] <= report["latency"]["latency_p95_ms"]


@pytest.mark.parametrize("tamper", ["change_weights", "add_file"])
def test_tampered_t1_folder_is_not_loaded(
    tmp_path: Path, mlflow_template: Path, tamper: str
) -> None:
    cfg = make_cfg(tmp_path, mlflow_template)
    model_dir = train_transformer(cfg).model_path
    if tamper == "change_weights":
        weights = model_dir / "model.safetensors"
        weights.write_bytes(weights.read_bytes() + b"x")
    else:
        (model_dir / "extra.py").write_text("print('not ours')\n")
    with pytest.raises(ModelChecksumError, match="hash differs"):
        evaluate_on_test(cfg, "t1")
    assert not (cfg.evaluation.reports_dir / "t1.json").exists()


@pytest.fixture
def export_dir(tmp_path: Path, mlflow_template: Path) -> tuple[Config, Path]:
    """A Colab-style export made on another 'machine' (own root) with the same frozen split."""
    colab_cfg = make_cfg(tmp_path / "colab", mlflow_template)
    out = tmp_path / "t1_export"
    export_t1_run(colab_cfg, out)
    return make_cfg(tmp_path / "local", mlflow_template), out


def test_colab_export_is_imported_as_full_run_and_evaluates(
    export_dir: tuple[Config, Path],
) -> None:
    cfg, out = export_dir
    assert (out / EXPORT_RECORD).exists() and not (out / "trainer").exists()
    result = import_t1_run(cfg, out)
    run = mlflow.get_run(result.run_id)
    assert run.info.run_name == "t1" and run.data.tags["mode"] == "full"
    assert run.data.tags["trained_on"] == "colab"
    m = run.data.metrics
    assert abs(m["val_intent_macro_f1"] - m["colab_val_intent_macro_f1"]) <= 0.005
    assert evaluate_on_test(cfg, "t1")["mlflow_run_id"] == result.run_id


def test_tampered_colab_export_is_refused(export_dir: tuple[Config, Path]) -> None:
    cfg, out = export_dir
    (out / "model" / "config.json").write_text("{}")
    with pytest.raises(ModelChecksumError, match="config.json"):
        import_t1_run(cfg, out)
    assert not (cfg.models_dir / "t1").exists()


def test_colab_export_from_another_split_is_refused(export_dir: tuple[Config, Path]) -> None:
    cfg, out = export_dir
    record = json.loads((out / EXPORT_RECORD).read_text())
    record["params"]["split_sha256_test"] = "0" * 64
    (out / EXPORT_RECORD).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="another test split"):
        import_t1_run(cfg, out)


def test_second_import_does_not_overwrite_the_first(export_dir: tuple[Config, Path]) -> None:
    cfg, out = export_dir
    import_t1_run(cfg, out)
    with pytest.raises(FileExistsError):
        import_t1_run(cfg, out)
