"""Train B0/B1/T1 on train, select on val, log everything to MLflow.

Never reads `test.csv`. One MLflow run per call; the model is saved to
`models/<run name>/` with a SHA-256 sidecar and logged to MLflow.
B0/B1: `model.pkl`, C chosen on val. T1: a `model/` folder, best epoch chosen on val.
T1 FULL runs on Colab (`export_t1_run`) and is logged here with `--import-dir`.
Run: `python -m supportrouter.train --model {b0,b1,t1} [--sample [N]]`
     `python -m supportrouter.train --model t1 --import-dir <unzipped Colab export>`.
"""

import argparse
import json
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
import pandas as pd
import sklearn
import torch
import transformers
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from supportrouter.config import Config, load_config
from supportrouter.data.load import MANIFEST_NAME, file_sha256, load_split
from supportrouter.data.split import sample_stratified
from supportrouter.evaluate import find_run, intent_and_category_metrics
from supportrouter.models import transformer as t1
from supportrouter.models.baseline import build_b0, build_b1
from supportrouter.predict import ModelChecksumError

Family = Literal["b0", "b1", "t1"]
EXPORT_RECORD = "run.json"  # written by `export_t1_run` on Colab, read by `import_t1_run`

# Names and value shapes that look like credentials. SHA-256 hex digests are allowed.
_SECRET_NAME = re.compile(r"key|secret|token|password|credential", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(sk-|sk_|AKIA|ghp_|gsk_|hf_|xox[bp]-)[A-Za-z0-9_\-]{8,}")


class SecretInParamsError(ValueError):
    """A parameter about to be logged looks like a credential."""


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_path: Path
    params: dict[str, Any]
    val_metrics: dict[str, float]


def check_no_secrets(params: dict[str, Any]) -> None:
    """Refuse to log params whose name or value looks like a key; reports names only."""
    bad = [k for k, v in params.items() if _SECRET_NAME.search(k) or _SECRET_VALUE.search(str(v))]
    if bad:
        raise SecretInParamsError(f"refusing to log key-like params: {sorted(bad)}")


def configure_mlflow(cfg: Config) -> str:
    """Point MLflow at the local SQLite store and return the experiment id."""
    cfg.mlflow.db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment)
    if experiment is not None:
        return str(experiment.experiment_id)
    return str(
        mlflow.create_experiment(
            cfg.mlflow.experiment, artifact_location=cfg.mlflow.artifact_dir.as_uri()
        )
    )


def _fit_and_score(
    model: DummyClassifier | Pipeline,
    train_xy: tuple[list[str], list[str]],
    val_xy: tuple[list[str], list[str]],
    cfg: Config,
) -> dict[str, float]:
    model.fit(*train_xy)
    val_x, val_y = val_xy
    return intent_and_category_metrics(val_y, list(model.predict(val_x)), cfg.labels)


def _run_name(family: Family, sample: int | None) -> str:
    return family if sample is None else f"{family}-sample{sample}"


def _load_train_val(
    cfg: Config, family: Family, sample: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Hash-checked train/val, the optional train sample, common params."""
    train = load_split(cfg, "train")
    val = load_split(cfg, "val")
    if sample is not None:
        train = sample_stratified(train, sample, cfg.seed)
    manifest = json.loads((cfg.split.processed_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    params: dict[str, Any] = {
        "family": family,
        "mode": "full" if sample is None else "sample",
        "sample_size": sample,
        "train_rows": len(train),
        "seed": cfg.seed,
        "input_column": "instruction",
        **{f"split_sha256_{n}": manifest["splits"][n]["sha256"] for n in ("train", "val", "test")},
        "data_revision": manifest["source"]["hf_revision"],
    }
    return train, val, params


def train_baseline(cfg: Config, family: Family, sample: int | None = None) -> TrainResult:
    """Fit B0 or B1 on train (optionally an N-row stratified sample) and log the run."""
    train, val, params = _load_train_val(cfg, family, sample)
    train_x, train_y = train["instruction"].tolist(), train["intent"].tolist()
    val_x, val_y = val["instruction"].tolist(), val["intent"].tolist()
    mode, run_name = params["mode"], _run_name(family, sample)

    grid: dict[float, dict[str, float]] = {}
    if family == "b0":
        model: DummyClassifier | Pipeline = build_b0()
        val_metrics = _fit_and_score(model, (train_x, train_y), (val_x, val_y), cfg)
        params["strategy"] = "most_frequent"
    else:
        best_c, best_model, val_metrics = 0.0, None, {}
        for c in cfg.baselines.c_grid:  # model selection on val only
            candidate = build_b1(cfg.baselines, c, cfg.seed)
            grid[c] = _fit_and_score(candidate, (train_x, train_y), (val_x, val_y), cfg)
            if best_model is None or grid[c]["intent_macro_f1"] > val_metrics["intent_macro_f1"]:
                best_c, best_model, val_metrics = c, candidate, grid[c]
        assert best_model is not None
        model = best_model
        params |= {
            "ngram_range": list(cfg.baselines.ngram_range),
            "c_grid": list(cfg.baselines.c_grid),
            "best_c": best_c,
            "max_iter": cfg.baselines.max_iter,
        }

    check_no_secrets(params)
    out_dir = cfg.models_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    model_path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    sidecar = model_path.with_name(model_path.name + ".sha256")
    sidecar.write_text(f"{file_sha256(model_path)}  {model_path.name}\n", encoding="utf-8")

    configure_mlflow(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"family": family, "mode": mode, "phase": "4"})
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        for i, (c, scores) in enumerate(grid.items()):
            mlflow.log_metric("grid_val_intent_macro_f1", scores["intent_macro_f1"], step=i)
            mlflow.log_metric(f"grid_val_intent_macro_f1_C_{c:g}", scores["intent_macro_f1"])
        mlflow.log_artifact(str(model_path), artifact_path="local_copy")
        mlflow.log_artifact(str(sidecar), artifact_path="local_copy")
        mlflow.log_dict({"intents": list(cfg.labels.intents)}, "labels.json")
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(val_x[:5], list(model.predict(val_x[:5]))),
            pip_requirements=[f"scikit-learn=={sklearn.__version__}"],
        )
    return TrainResult(run.info.run_id, model_path, params, val_metrics)


def _fit_and_save_t1(cfg: Config, sample: int | None, out_dir: Path) -> dict[str, Any]:
    """Fine-tune T1 on train, keep the best val epoch, save `out_dir/model/` + `model.sha256`.

    Shared by the local run and the Colab export (one code path).
    Returns the run record: params, per-epoch history, val metrics, training seconds.
    """
    train, val, params = _load_train_val(cfg, "t1", sample)
    val_x, val_y = val["instruction"].tolist(), val["intent"].tolist()
    tcfg = cfg.transformer
    params |= {
        "model_name": tcfg.model_name,
        "model_revision": tcfg.revision,
        "max_length": tcfg.max_length,
        "learning_rate": tcfg.learning_rate,
        "epochs": tcfg.epochs,
        "batch_size": tcfg.batch_size,
        "warmup_ratio": tcfg.warmup_ratio,
        "weight_decay": tcfg.weight_decay,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    check_no_secrets(params)  # before any slow work

    tokenizer, model = t1.load_pretrained(tcfg, cfg.labels.intents)
    start = time.perf_counter()
    model, history = t1.fit_t1(
        model,
        tokenizer,
        (train["instruction"].tolist(), train["intent"].tolist()),
        (val_x, val_y),
        tcfg,
        cfg.seed,
        out_dir / "trainer",
    )
    train_seconds = time.perf_counter() - start
    shutil.rmtree(out_dir / "trainer")  # per-epoch checkpoints; the best one is in `model`
    t1.save_t1(model, tokenizer, out_dir / "model")

    val_pred, _ = t1.predict_with_scores(model, tokenizer, val_x, tcfg.max_length)
    epochs = [h for h in history if "eval_macro_f1" in h]
    params["best_epoch"] = int(max(epochs, key=lambda h: h["eval_macro_f1"])["epoch"])
    return {
        "params": params,
        "history": history,
        "val_metrics": intent_and_category_metrics(val_y, val_pred, cfg.labels),
        "train_seconds": train_seconds,
    }


def _log_t1_run(
    cfg: Config,
    run_name: str,
    record: dict[str, Any],
    model_dir: Path,
    trained_on: str,
    extra_metrics: dict[str, float] | None = None,
) -> TrainResult:
    """One MLflow run for a saved T1 folder: params, val metrics, curves, model + checksum."""
    params, history = record["params"], record["history"]
    check_no_secrets(params)
    sidecar = model_dir.with_name(model_dir.name + ".sha256")
    configure_mlflow(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name=run_name) as run:
        tags = {"family": "t1", "mode": params["mode"], "phase": "5", "trained_on": trained_on}
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in record["val_metrics"].items()})
        mlflow.log_metrics({"train_seconds": record["train_seconds"], **(extra_metrics or {})})
        for h in history:  # learning curves, one point per epoch
            step = int(h["epoch"])
            if "loss" in h:
                mlflow.log_metric("epoch_train_loss", h["loss"], step=step)
            if "eval_loss" in h:
                mlflow.log_metric("epoch_val_loss", h["eval_loss"], step=step)
                mlflow.log_metric("epoch_val_macro_f1", h["eval_macro_f1"], step=step)
        mlflow.log_dict({"history": history}, "history.json")
        mlflow.log_dict({"intents": list(cfg.labels.intents)}, "labels.json")
        mlflow.log_artifact(str(sidecar), artifact_path="local_copy")
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    return TrainResult(run.info.run_id, model_dir, params, record["val_metrics"])


def train_transformer(cfg: Config, sample: int | None = None) -> TrainResult:
    """Local T1 run (CPU SAMPLE in practice): fit, save to `models/<run>/`, log to MLflow."""
    run_name = _run_name("t1", sample)
    record = _fit_and_save_t1(cfg, sample, cfg.models_dir / run_name)
    return _log_t1_run(cfg, run_name, record, cfg.models_dir / run_name / "model", "local")


def export_t1_run(cfg: Config, out_dir: Path, sample: int | None = None) -> Path:
    """Colab side of the T1 FULL import: fit and save into `out_dir`, write `run.json`.

    No MLflow here; `import_t1_run` logs the run locally after checking every hash.
    """
    record = _fit_and_save_t1(cfg, sample, out_dir)
    path = out_dir / EXPORT_RECORD
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def import_t1_run(cfg: Config, export_dir: Path) -> TrainResult:
    """Local side of the T1 FULL import: verify a Colab export, re-score it on val, log it.

    Refused when a file hash differs, the run used another split or model revision, the
    destination already exists, or the local val macro-F1 differs by more than the tolerance.
    """
    record = json.loads((export_dir / EXPORT_RECORD).read_text(encoding="utf-8"))
    params = record["params"]
    check_no_secrets(params)
    sidecar_text = (export_dir / "model.sha256").read_text(encoding="utf-8")
    bad = t1.checksum_mismatches(export_dir / "model", sidecar_text)
    if bad:
        raise ModelChecksumError(f"{bad} hash differs from the export's model.sha256")
    manifest = json.loads((cfg.split.processed_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    for name in ("train", "val", "test"):
        if params[f"split_sha256_{name}"] != manifest["splits"][name]["sha256"]:
            raise ValueError(f"export was trained on another {name} split version")
    if params["model_revision"] != cfg.transformer.revision:
        raise ValueError(f"export used model revision {params['model_revision'][:12]}")

    run_name = _run_name("t1", params["sample_size"])
    dest = cfg.models_dir / run_name
    if dest.exists():
        raise FileExistsError(f"{dest} exists; move it away before importing another run")
    shutil.copytree(export_dir / "model", dest / "model")
    shutil.copy(export_dir / "model.sha256", dest / "model.sha256")

    val = load_split(cfg, "val")
    tokenizer, model = t1.load_t1(dest / "model")
    val_pred, _ = t1.predict_with_scores(
        model, tokenizer, val["instruction"].tolist(), cfg.transformer.max_length
    )
    local = intent_and_category_metrics(val["intent"].tolist(), val_pred, cfg.labels)
    remote = record["val_metrics"]
    gap = abs(local["intent_macro_f1"] - remote["intent_macro_f1"])
    if gap > cfg.transformer.import_val_tolerance:
        shutil.rmtree(dest)  # our own copy from this call; nothing else is removed
        raise ValueError(f"local val macro-F1 differs from the export by {gap:.4f}")
    record = {**record, "val_metrics": local}  # logged val metrics are the local ones
    colab = {f"colab_val_{k}": v for k, v in remote.items()}
    return _log_t1_run(cfg, run_name, record, dest / "model", "colab", colab)


# T1 FULL is served (`champion`); B1 FULL is the documented fallback (`fallback`).
SERVED_ALIASES: dict[str, str] = {"champion": "t1", "fallback": "b1"}


def register_served_models(cfg: Config, aliases: dict[str, str] = SERVED_ALIASES) -> dict[str, str]:
    """Register the FULL run of each family and point its alias at it.

    Idempotent: a run that already has a version is reused. B1 was logged with
    `mlflow.sklearn.log_model`; T1 is a plain `model/` artifact folder, so its version is
    created from that folder (MLflow 3 `register_model` only finds logged models).
    Returns alias -> version.
    """
    configure_mlflow(cfg)
    client = mlflow.MlflowClient()
    name = cfg.mlflow.registered_model
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name, description="SupportRouter intent classifier")
    result: dict[str, str] = {}
    for alias, family in aliases.items():
        run = find_run(cfg, family, None)
        existing = client.search_model_versions(f"name = '{name}' and run_id = '{run.info.run_id}'")
        if existing:
            version = existing[0].version
        elif family == "t1":
            source = f"{run.info.artifact_uri}/model"
            version = client.create_model_version(name, source, run_id=run.info.run_id).version
        else:
            version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name).version
        client.set_model_version_tag(name, version, "family", family)
        client.set_model_version_tag(name, version, "decision", "served T1, fallback B1")
        client.set_registered_model_alias(name, alias, version)
        result[alias] = str(version)
    return result


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["b0", "b1", "t1"])
    parser.add_argument("--sample", type=int, nargs="?", const=cfg.sample_size, default=None)
    parser.add_argument("--import-dir", type=Path, default=None, help="T1 Colab export")
    parser.add_argument("--register", action="store_true", help="set registry aliases")
    args = parser.parse_args()
    if args.register:
        print(json.dumps(register_served_models(cfg)))
        return
    if args.model is None:
        parser.error("--model is required unless --register is given")
    if args.import_dir is not None:
        if args.model != "t1":
            parser.error("--import-dir is only for --model t1")
        result = import_t1_run(cfg, args.import_dir)
    elif args.model == "t1":
        result = train_transformer(cfg, sample=args.sample)
    else:
        result = train_baseline(cfg, args.model, sample=args.sample)
    summary = {
        "run_id": result.run_id,
        "model_path": str(result.model_path.relative_to(cfg.models_dir.parent)),
        "best_c": result.params.get("best_c"),
        "best_epoch": result.params.get("best_epoch"),
        "val": result.val_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
