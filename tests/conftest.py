"""Shared fixtures: a tiny frozen split with its own MLflow store and models dir."""

import dataclasses
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from supportrouter.config import Config, load_config
from supportrouter.data.split import write_frozen_splits

WORDS = {
    "cancel_order": ("ORDER", "cancel stop order purchase"),
    "track_refund": ("REFUND", "refund money back status"),
    "get_invoice": ("INVOICE", "invoice bill receipt copy"),
}


def split_frame(n: int, offset: int) -> pd.DataFrame:
    rows = []
    for intent, (category, words) in WORDS.items():
        w = words.split()
        for i in range(offset, offset + n):
            text = f"{w[i % 4]} {w[(i + 1) % 4]} please {i}"
            rows.append(("B", text, category, intent, "unused"))
    return pd.DataFrame(rows, columns=["flags", "instruction", "category", "intent", "response"])


def make_cfg(root: Path, template_db: Path) -> Config:
    """Default config pointed at `root` for splits, MLflow, models and reports.

    The MLflow database is copied from an already-migrated template: creating a fresh
    SQLite store runs MLflow's schema migrations, which takes 10+ seconds each time.
    """
    (root / "mlruns").mkdir(parents=True)
    shutil.copy(template_db, root / "mlruns" / "mlflow.db")
    splits = {"train": split_frame(20, 0), "val": split_frame(6, 100), "test": split_frame(6, 200)}
    write_frozen_splits(splits, root / "processed", {"source": {"hf_revision": "abc"}})
    base = load_config()
    return dataclasses.replace(
        base,
        split=dataclasses.replace(base.split, processed_dir=root / "processed"),
        mlflow=dataclasses.replace(
            base.mlflow,
            db_path=root / "mlruns" / "mlflow.db",
            artifact_dir=root / "mlruns" / "a",
            server_uri=None,  # tests never follow an MLflow server URL from the environment
        ),
        models_dir=root / "models",
        evaluation=dataclasses.replace(
            base.evaluation, reports_dir=root / "reports", predictions_dir=root / "predictions"
        ),
        llm=dataclasses.replace(base.llm, results_path=root / "llm_zero_shot.csv"),
        api=dataclasses.replace(base.api, log_path=root / "logs" / "predictions.jsonl"),
        quality_gate=dataclasses.replace(
            base.quality_gate, report_path=root / "reports" / "quality_gate.json"
        ),
        monitoring=dataclasses.replace(
            base.monitoring,
            batches_path=root / "logs" / "traffic_batches.json",
            report_dir=root / "reports" / "drift",
        ),
    )


@pytest.fixture(scope="session")
def mlflow_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("template") / "mlflow.db"
    mlflow.MlflowClient(tracking_uri=f"sqlite:///{db}").search_experiments()  # runs migrations
    return db


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: pytest.TempPathFactory, mlflow_template: Path) -> Config:
    return make_cfg(tmp_path_factory.mktemp("run"), mlflow_template)
