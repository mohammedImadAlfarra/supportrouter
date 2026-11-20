from pathlib import Path

import mlflow
import pandas as pd
import pytest

from supportrouter import train as train_module
from supportrouter.config import Config
from supportrouter.data.load import file_sha256
from supportrouter.train import (
    SecretInParamsError,
    TrainResult,
    check_no_secrets,
    train_baseline,
)


@pytest.fixture(scope="module")
def b1_runs(cfg: Config) -> tuple[TrainResult, TrainResult]:
    return train_baseline(cfg, "b1"), train_baseline(cfg, "b1")


def test_same_seed_gives_same_val_metrics(
    cfg: Config, b1_runs: tuple[TrainResult, TrainResult]
) -> None:
    first, second = b1_runs
    assert first.run_id != second.run_id
    for name, value in first.val_metrics.items():
        assert abs(value - second.val_metrics[name]) <= cfg.baselines.metric_tolerance


def test_training_reads_only_train_and_val(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real = train_module.load_split

    def spy(c: Config, name: str) -> pd.DataFrame:
        seen.append(name)
        return real(c, name)

    monkeypatch.setattr(train_module, "load_split", spy)
    train_baseline(cfg, "b0")
    assert seen == ["train", "val"]


def test_logged_params_contain_no_key_like_strings(
    b1_runs: tuple[TrainResult, TrainResult],
) -> None:
    logged = mlflow.get_run(b1_runs[0].run_id).data.params
    assert "split_sha256_test" in logged  # 64-char hex digests are allowed
    check_no_secrets(dict(logged))


@pytest.mark.parametrize(
    "params",
    [{"api_key": "x"}, {"provider": "sk-ant-abcdef123456"}, {"note": "ghp_abcdefghijklmnop"}],
)
def test_key_like_params_are_refused(params: dict[str, str]) -> None:
    with pytest.raises(SecretInParamsError, match="refusing to log key-like params"):
        check_no_secrets(params)


def test_b1_logs_best_c_from_the_val_grid(
    cfg: Config, b1_runs: tuple[TrainResult, TrainResult]
) -> None:
    run = mlflow.get_run(b1_runs[0].run_id).data
    grid = {c: run.metrics[f"grid_val_intent_macro_f1_C_{c:g}"] for c in cfg.baselines.c_grid}
    best = max(grid.values())
    first_best = next(c for c, v in grid.items() if v == best)  # ties keep the smaller C
    assert float(run.params["best_c"]) == first_best
    assert run.metrics["val_intent_macro_f1"] == best
    assert run.params["mode"] == "full"


def test_model_file_matches_its_sha256_sidecar(b1_runs: tuple[TrainResult, TrainResult]) -> None:
    path: Path = b1_runs[0].model_path
    digest, name = path.with_name("model.pkl.sha256").read_text().split()
    assert (digest, name) == (file_sha256(path), "model.pkl")


def test_sample_mode_trains_on_n_rows(cfg: Config) -> None:
    result = train_baseline(cfg, "b0", sample=30)
    assert result.params["mode"] == "sample"
    assert result.params["train_rows"] == 30
    assert result.model_path.parent.name == "b0-sample30"


def test_registry_aliases_point_at_the_runs_and_rerun_is_idempotent(cfg: Config) -> None:
    import mlflow

    from supportrouter.train import register_served_models, train_baseline

    b0, b1 = train_baseline(cfg, "b0"), train_baseline(cfg, "b1")
    aliases = {"champion": "b1", "fallback": "b0"}  # sklearn stand-ins; T1 path runs for real
    first = register_served_models(cfg, aliases)
    assert first == register_served_models(cfg, aliases)  # no new versions on re-run
    client = mlflow.MlflowClient()
    name = cfg.mlflow.registered_model
    assert client.get_model_version_by_alias(name, "champion").run_id == b1.run_id
    assert client.get_model_version_by_alias(name, "fallback").run_id == b0.run_id
    champion_tags = client.get_model_version_by_alias(name, "champion").tags
    assert champion_tags["decision"] == "served T1, fallback B1"
