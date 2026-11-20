import dataclasses
import json
from pathlib import Path

import mlflow
import pytest

from conftest import make_cfg
from supportrouter.config import Config
from supportrouter.data.load import SplitHashError, file_sha256
from supportrouter.evaluate import (
    classification_metrics,
    evaluate_on_test,
    per_intent_f1,
)
from supportrouter.predict import ModelChecksumError
from supportrouter.train import train_baseline


@pytest.fixture(scope="module")
def trained(cfg: Config) -> Config:
    train_baseline(cfg, "b0")
    train_baseline(cfg, "b1")
    return cfg


def test_macro_f1_counts_never_predicted_classes_as_zero() -> None:
    # majority-class predictions: F1(a) = 2*0.5*1/1.5 = 2/3, F1(b) = 0, macro = 1/3
    m = classification_metrics(["a", "a", "b", "b"], ["a"] * 4, labels=["a", "b"])
    assert m["accuracy"] == 0.5
    assert m["macro_f1"] == pytest.approx(1 / 3)


def test_per_intent_f1_lists_every_intent(trained: Config) -> None:
    scores = per_intent_f1(["cancel_order"], ["cancel_order"], trained.labels)
    assert len(scores) == 27 and scores["cancel_order"] == 1.0 and scores["get_refund"] == 0.0


def test_report_written_and_logged_to_the_run(trained: Config) -> None:
    report = evaluate_on_test(trained, "b1")
    saved = json.loads((trained.evaluation.reports_dir / "b1.json").read_text())
    assert saved == report
    assert report["test_rows"] == 18 and report["mode"] == "full"
    assert report["latency"]["latency_p50_ms"] <= report["latency"]["latency_p95_ms"]
    assert report["model_size_bytes"] > 0
    logged = mlflow.get_run(report["mlflow_run_id"]).data.metrics
    assert logged["test_intent_macro_f1"] == report["metrics"]["intent_macro_f1"]


def test_test_hash_unchanged_across_evaluations(trained: Config) -> None:
    test_csv = trained.split.processed_dir / "test.csv"
    before = file_sha256(test_csv)
    first, second = evaluate_on_test(trained, "b0"), evaluate_on_test(trained, "b0")
    assert file_sha256(test_csv) == before == first["test_sha256"] == second["test_sha256"]
    assert first["metrics"] == second["metrics"]


def test_modified_test_file_stops_evaluation(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = make_cfg(tmp_path, mlflow_template)
    train_baseline(cfg, "b0")
    test_csv = cfg.split.processed_dir / "test.csv"
    test_csv.write_text(test_csv.read_text().replace("please 200", "please 999"))
    with pytest.raises(SplitHashError, match="test.csv"):
        evaluate_on_test(cfg, "b0")
    assert not (cfg.evaluation.reports_dir / "b0.json").exists()


def test_tampered_model_is_not_loaded(tmp_path: Path, mlflow_template: Path) -> None:
    cfg = make_cfg(tmp_path, mlflow_template)
    result = train_baseline(cfg, "b0")
    result.model_path.write_bytes(result.model_path.read_bytes() + b"x")
    with pytest.raises(ModelChecksumError, match="hash differs"):
        evaluate_on_test(cfg, "b0")


def test_subset_report_is_separate_and_keeps_full_test_metrics(trained: Config) -> None:
    cfg = dataclasses.replace(trained, llm=dataclasses.replace(trained.llm, subset_size=9))
    full = evaluate_on_test(cfg, "b1")
    sub = evaluate_on_test(cfg, "b1", subset=True)
    assert (sub["eval_set"], sub["test_rows"], full["eval_set"]) == ("test-subset-9", 9, "test")
    assert (cfg.evaluation.reports_dir / "b1-subset9.json").exists()
    logged = mlflow.get_run(full["mlflow_run_id"]).data.metrics
    assert logged["test_intent_macro_f1"] == full["metrics"]["intent_macro_f1"]
    assert logged["subset_test_intent_macro_f1"] == sub["metrics"]["intent_macro_f1"]


def test_predictions_file_has_no_text_and_matches_test(trained: Config) -> None:
    import pandas as pd

    from supportrouter.evaluate import write_test_predictions

    out = write_test_predictions(trained, "b1")
    rows = pd.read_csv(out)
    assert list(rows.columns) == [
        "test_row",
        "flags",
        "intent",
        "prediction",
        "score",
        "near_dup_cosine",
    ]
    assert len(rows) == 18 and "instruction" not in rows.columns
    assert rows["near_dup_cosine"].between(0, 1.000001).all()


def test_comparison_numbers_come_from_the_metrics_files(trained: Config) -> None:
    from supportrouter.evaluate import build_comparison

    full = evaluate_on_test(trained, "b1")
    table = build_comparison(trained)
    row = next(line for line in table.splitlines() if line.startswith("| B1 TF-IDF+LR FULL"))
    assert f"{full['metrics']['intent_macro_f1']:,.4f}" in row
    assert f"{full['latency']['latency_p95_ms']:,.3f}" in row
    assert full["mlflow_run_id"][:8] in row
    assert "## Evaluation set: `test` (18 messages)" in table


def test_error_analysis_counts_pairs_and_draws_figure(trained: Config) -> None:
    import pandas as pd

    from supportrouter.evaluate import confused_pairs, write_error_analysis, write_test_predictions

    rows = pd.DataFrame(
        {
            "intent": ["a", "a", "a", "b"],
            "prediction": ["b", "b", "a", "c"],
            "test_row": [0, 1, 2, 3],
        }
    )
    assert confused_pairs(rows, 5).values.tolist() == [["a", "b", 2], ["b", "c", 1]]
    write_test_predictions(trained, "b1")
    out = write_error_analysis(trained)
    assert "## B1 FULL (test, 18)" in out.read_text()
    assert (out.parent / "figures" / "confusion_b1.png").stat().st_size > 0


def test_robustness_slices_count_rows_containing_each_flag_letter() -> None:
    import pandas as pd

    from supportrouter.evaluate import robustness_table

    rows = pd.DataFrame(
        {
            "flags": ["BZ", "BL", "BLZ", "B"],
            "intent": ["a", "a", "b", "b"],
            "prediction": ["b", "a", "b", "b"],
            "near_dup_cosine": [0.95, 0.5, 0.91, 0.2],
        }
    )
    table = robustness_table(rows, 0.90).set_index("slice")
    assert table.loc["flag Z", ["rows", "errors"]].tolist() == [2, 1]
    assert table.loc["flag L", "accuracy"] == 1.0
    assert table.loc["near-duplicate in train (cosine >= 0.90)", "rows"] == 2
    assert table.loc["all", "accuracy"] == 0.75


def test_typo_perturbation_is_seeded_and_keeps_placeholders() -> None:
    import random

    from supportrouter.evaluate import perturb_text

    text = "please cancel purchase {{Order Number}} quickly"
    first = perturb_text(text, 1.0, random.Random(42), 4)
    assert first == perturb_text(text, 1.0, random.Random(42), 4)
    assert first != text and "{{Order Number}}" in first
    assert perturb_text(text, 0.0, random.Random(42), 4) == text
    assert perturb_text("ok an it", 1.0, random.Random(42), 4) == "ok an it"  # short words kept


def test_coverage_curve_counts_auto_routed_and_caught_errors() -> None:
    from supportrouter.evaluate import coverage_curve

    curve = coverage_curve(["a", "a", "b", "b"], ["a", "b", "b", "a"], [0.99, 0.55, 0.85, 0.65])
    at = {r["threshold"]: r for r in curve}
    assert at[0.5]["coverage"] == 1.0 and at[0.5]["auto_errors"] == 2
    assert at[0.7] == {
        "threshold": 0.7,
        "coverage": 0.5,
        "auto_accuracy": 1.0,
        "auto_errors": 0,
        "to_review": 2,
        "errors_caught": 2,
    }
