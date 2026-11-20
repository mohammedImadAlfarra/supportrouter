import dataclasses
import json
from pathlib import Path
from types import MappingProxyType

import mlflow
import pandas as pd
import pytest

from conftest import WORDS, make_cfg
from supportrouter.config import Config, LabelConfig
from supportrouter.data.load import SplitHashError
from supportrouter.evaluate import evaluate_llm, llm_metrics
from supportrouter.models.llm_adapter import CostCapError
from supportrouter.models.llm_zero_shot import INVALID


def llm_cfg(tmp_path: Path, template: Path, **llm: object) -> Config:
    """Tiny split (3 intents x 6 test rows), MockLLM, results in tmp; labels cut to 3 intents
    so L2 can pick 2 train examples per intent from the tiny train split."""
    cfg = make_cfg(tmp_path, template)
    labels = LabelConfig(MappingProxyType({i: c for i, (c, _) in WORDS.items()}))
    settings = {"provider": "mock", "subset_size": 9, "results_path": tmp_path / "llm.csv", **llm}
    return dataclasses.replace(cfg, labels=labels, llm=dataclasses.replace(cfg.llm, **settings))


def test_invalid_prediction_is_wrong_for_intent_and_category(cfg: Config) -> None:
    m = llm_metrics(["cancel_order", "track_refund"], ["cancel_order", INVALID], cfg.labels)
    assert m["intent_accuracy"] == 0.5 and m["category_accuracy"] == 0.5


@pytest.mark.parametrize("condition", ["l1", "l2"])
def test_llm_run_writes_rows_report_and_mlflow_run(
    tmp_path: Path, mlflow_template: Path, condition: str
) -> None:
    cfg = llm_cfg(tmp_path, mlflow_template)
    report = evaluate_llm(cfg, condition)
    rows = pd.read_csv(cfg.llm.results_path)
    assert len(rows) == 9 and set(rows["condition"]) == {condition}
    expected = {"test_row", "intent", "prediction", "valid", "invalid_reason", "input_tokens"}
    assert expected <= set(rows.columns) and {"cost_usd", "latency_ms"} <= set(rows.columns)
    assert report["mode"] == "test-subset-9" and report["llm"]["total_cost_usd"] == 0.0
    saved = json.loads((cfg.evaluation.reports_dir / f"{report['model']}.json").read_text())
    assert saved == report
    run = mlflow.get_run(report["mlflow_run_id"])
    assert run.data.params["prompt_version"] == "v1"
    prompt = mlflow.artifacts.load_text(f"{run.info.artifact_uri}/prompt.txt")
    assert ("Examples:" in prompt) == (condition == "l2")


def test_second_condition_keeps_the_first_in_the_csv(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = llm_cfg(tmp_path, mlflow_template)
    evaluate_llm(cfg, "l1")
    evaluate_llm(cfg, "l2")
    evaluate_llm(cfg, "l1")  # a re-run replaces its own rows only
    counts = pd.read_csv(cfg.llm.results_path)["condition"].value_counts().to_dict()
    assert counts == {"l1": 9, "l2": 9}


def test_modified_test_file_stops_the_llm_run(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = llm_cfg(tmp_path, mlflow_template)
    test_csv = cfg.split.processed_dir / "test.csv"
    test_csv.write_text(test_csv.read_text().replace("please 200", "please 999"))
    with pytest.raises(SplitHashError, match="test.csv"):
        evaluate_llm(cfg, "l1")
    assert not cfg.llm.results_path.exists()


def test_paid_estimate_aborts_before_any_file_or_run(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = llm_cfg(tmp_path, mlflow_template, price_in_per_1m=0.10, price_out_per_1m=0.50)
    with pytest.raises(CostCapError):
        evaluate_llm(cfg, "l1")
    assert not cfg.llm.results_path.exists()
    assert not cfg.evaluation.reports_dir.exists()


def test_smoke_uses_val_and_writes_nothing(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = llm_cfg(tmp_path, mlflow_template)
    report = evaluate_llm(cfg, "l1", smoke=6)
    assert report["mode"] == "val-subset-6" and "val_sha256" in report
    assert "mlflow_run_id" not in report
    assert not cfg.llm.results_path.exists() and not cfg.evaluation.reports_dir.exists()
