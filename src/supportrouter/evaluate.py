"""Metrics shared by training (val) and final evaluation on the frozen test split.

Macro-F1 over all 27 intents is the primary metric. Category metrics come
from the predicted intent through the label map, so the two can never disagree.
This is the only module that reads `test.csv`, always through the hash check.
B0/B1 (`model.pkl`) and T1 (a `model/` folder) are loaded behind one `predict` callable.
LLM L1/L2 (Phase 6) run on a fixed seeded test subset; `--smoke N` uses val and writes nothing.
Run: `python -m supportrouter.evaluate --model {b0,b1,t1,l1,l2} [--sample [N]] [--smoke N]`.
"""

import argparse
import json
import os
import platform
import random
import re
import string
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from supportrouter.config import Config, LabelConfig, load_config
from supportrouter.data.load import file_sha256, load_split
from supportrouter.data.split import nearest_train_cosine
from supportrouter.models import llm_zero_shot as llm
from supportrouter.models.llm_adapter import check_cost_cap, make_client
from supportrouter.predict import (  # model loading lives in predict.py
    PredictFn,
    load_checked_model,
)


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    """Accuracy, macro-F1 and weighted-F1.

    `labels=` lists every class, so a class the model never predicts still counts with F1 0;
    `zero_division=0` makes that explicit instead of a warning.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
        ),
    }


def intent_and_category_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: LabelConfig
) -> dict[str, float]:
    """Intent metrics plus category metrics derived from the same predictions."""
    to_category = labels.intent_to_category
    intent = classification_metrics(y_true, y_pred, labels.intents)
    category = classification_metrics(
        [to_category[i] for i in y_true], [to_category[i] for i in y_pred], labels.categories
    )
    return {
        **{f"intent_{k}": v for k, v in intent.items()},
        **{f"category_{k}": v for k, v in category.items()},
    }


def per_intent_f1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: LabelConfig
) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=list(labels.intents), average=None, zero_division=0)
    return {intent: float(f) for intent, f in zip(labels.intents, scores, strict=True)}


def find_run(cfg: Config, family: str, sample: int | None) -> mlflow.entities.Run:
    """Latest MLflow run of this family and mode (the model whose configuration is final)."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mode = "full" if sample is None else "sample"
    query = f"tags.family = '{family}' and tags.mode = '{mode}'"
    if sample is not None:
        query += f" and params.sample_size = '{sample}'"
    runs = mlflow.search_runs(
        experiment_names=[cfg.mlflow.experiment],
        filter_string=query,
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise LookupError(f"no MLflow run for {family} ({mode}); train it first")
    return runs[0]


def measure_latency(predict: PredictFn, texts: Sequence[str], cfg: Config) -> dict[str, float]:
    """Single-message predict on CPU, warm: discard warm-up calls, then time each call."""
    calls = cfg.evaluation.latency_calls
    pool = [texts[i % len(texts)] for i in range(cfg.evaluation.latency_warmup + calls)]
    for text in pool[: cfg.evaluation.latency_warmup]:
        predict([text])
    timings = []
    for text in pool[cfg.evaluation.latency_warmup :]:
        start = time.perf_counter()
        predict([text])
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "latency_calls": calls,
    }


# Selected hyperparameters copied into the report (B1: C; T1: epoch and model pin).
_REPORT_PARAMS = ("best_c", "c_grid", "best_epoch", "epochs", "max_length", "model_revision")


def _hardware() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version()}


def write_test_predictions(cfg: Config, family: str, sample: int | None = None) -> Path:
    """Per-row predictions of the final model on test for error and robustness analysis (Phase 7).

    Columns: test_row, flags, intent, prediction, score, near_dup_cosine. No message text, so
    the file can be committed; texts are looked up from the hash-checked test split when needed.
    """
    run = find_run(cfg, family, sample)
    model = load_checked_model(cfg, run)
    test = load_split(cfg, "test")  # hash check
    if file_sha256(cfg.split.processed_dir / "test.csv") != run.data.params["split_sha256_test"]:
        raise ValueError(f"run {run.info.run_id[:8]} was trained on another split version")
    y_pred, scores = model.predict(test["instruction"].tolist())
    train = load_split(cfg, "train")
    frame = pd.DataFrame(
        {
            "test_row": range(len(test)),
            "flags": test["flags"].to_numpy(),
            "intent": test["intent"].to_numpy(),
            "prediction": y_pred,
            "score": np.round(scores, 6),
            "near_dup_cosine": np.round(
                nearest_train_cosine(train["instruction"], test["instruction"]), 6
            ),
        }
    )
    out = cfg.evaluation.predictions_dir / f"{run.info.run_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def evaluate_on_test(
    cfg: Config, family: str, sample: int | None = None, *, subset: bool = False
) -> dict[str, Any]:
    """Score the final model on test once, write `reports/metrics/<run>.json`, log `test_*`.

    `subset=True`: the same seeded 300-message subset as L1/L2,
    written to `<run>-subset<N>.json` and logged as `subset_test_*`, so the full-test numbers
    of the run stay untouched.
    """
    run = find_run(cfg, family, sample)
    model = load_checked_model(cfg, run)
    test = load_split(cfg, "test")  # raises SplitHashError if test.csv changed
    test_sha256 = file_sha256(cfg.split.processed_dir / "test.csv")
    if test_sha256 != run.data.params["split_sha256_test"]:
        raise ValueError(f"run {run.info.run_id[:8]} was trained on another split version")
    if subset:
        test = llm.select_subset(test, cfg.llm.subset_size, cfg.seed)
    texts, y_true = test["instruction"].tolist(), test["intent"].tolist()
    y_pred, _ = model.predict(texts)
    metrics = intent_and_category_metrics(y_true, y_pred, cfg.labels)
    latency = measure_latency(model.predict, texts, cfg)

    report: dict[str, Any] = {
        "model": run.info.run_name,
        "family": family,
        "mlflow_run_id": run.info.run_id,
        "mode": run.data.params["mode"],
        "eval_set": f"test-subset-{len(test)}" if subset else "test",
        "sample_size": sample,
        "seed": cfg.seed,
        "test_sha256": test_sha256,
        "test_rows": len(test),
        "params": {k: run.data.params[k] for k in _REPORT_PARAMS if k in run.data.params},
        "metrics": metrics,
        "per_intent_f1": per_intent_f1(y_true, y_pred, cfg.labels),
        "latency": latency,
        "model_size_bytes": model.size_bytes,
        "hardware": _hardware(),
    }
    suffix, prefix = (f"-subset{len(test)}", "subset_") if subset else ("", "")
    out = cfg.evaluation.reports_dir / f"{run.info.run_name}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({f"{prefix}test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in latency.items()})
        mlflow.log_artifact(str(out), artifact_path="reports")
    return report


def llm_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: LabelConfig
) -> dict[str, float]:
    """Like `intent_and_category_metrics`, but an invalid answer (`llm.INVALID`) has no category:
    it stays INVALID, so it is wrong for both intent and category (counted, not guessed)."""
    to_category = {**labels.intent_to_category, llm.INVALID: llm.INVALID}
    intent = classification_metrics(y_true, y_pred, labels.intents)
    category = classification_metrics(
        [to_category[i] for i in y_true], [to_category[i] for i in y_pred], labels.categories
    )
    return {
        **{f"intent_{k}": v for k, v in intent.items()},
        **{f"category_{k}": v for k, v in category.items()},
    }


def _write_llm_rows(path: Path, rows: pd.DataFrame) -> None:
    """Replace this condition's rows in `reports/llm_zero_shot.csv`, keep the other condition."""
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[old["condition"] != rows["condition"].iloc[0]], rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort_values(["condition", "test_row"]).to_csv(path, index=False)


def evaluate_llm(cfg: Config, condition: str, smoke: int | None = None) -> dict[str, Any]:
    """L1 (zero-shot) or L2 (few-shot) on the fixed seeded test subset.

    Order: hash-checked splits -> fixed prompt -> printed estimate + cap -> pinned
    client -> one warm-up call -> every message -> metrics, CSV rows, JSON, MLflow run.
    `smoke`: N val messages, printed only, so the pipeline is checked without touching test.
    """
    from supportrouter.train import check_no_secrets, configure_mlflow  # train imports us

    if condition not in ("l1", "l2"):
        raise ValueError(f"condition must be l1 or l2, got {condition!r}")
    labels, lcfg = cfg.labels.intents, cfg.llm
    examples = None
    if condition == "l2":
        train = load_split(cfg, "train")  # few-shot examples come from train only
        examples = llm.select_few_shot(train, labels, lcfg.few_shot_per_intent, cfg.seed)
    system = llm.build_system_prompt(labels, examples)
    source = "val" if smoke else "test"
    frame = load_split(cfg, source)  # hash check (for test)
    subset = llm.select_subset(frame, smoke or lcfg.subset_size, cfg.seed)

    estimate = llm.estimate_run(system, subset["instruction"].tolist(), lcfg)
    print(json.dumps({"estimate": estimate, "cap_usd": lcfg.cost_cap_usd}), file=sys.stderr)
    check_cost_cap(estimate["usd"], lcfg.cost_cap_usd)  # aborts before the first call

    client = make_client(lcfg, labels, cfg.seed)
    client.complete(system, "warm-up", max_tokens=lcfg.max_output_tokens)  # load model, not timed
    rows = llm.run_condition(client, condition, system, subset, labels, lcfg)
    y_true, y_pred = rows["intent"].tolist(), rows["prediction"].tolist()
    stats = llm.run_stats(rows)
    report: dict[str, Any] = {
        "model": f"{condition}-{client.name}-{re.sub(r'[^a-z0-9]+', '-', lcfg.model.lower())}",
        "family": condition,
        "mode": f"{source}-subset-{len(subset)}",
        "seed": cfg.seed,
        "test_sha256" if source == "test" else "val_sha256": file_sha256(
            cfg.split.processed_dir / f"{source}.csv"
        ),
        "test_rows": len(subset),
        "params": {
            "provider": client.name,
            "llm_model": lcfg.model,
            "model_digest": lcfg.model_digest,
            "prompt_version": llm.PROMPT_VERSION,
            "prompt_chars": len(system),
            "few_shot_per_intent": lcfg.few_shot_per_intent if examples else 0,
            "temperature": 0.0,
            "max_retries": lcfg.max_retries,
        },
        "metrics": llm_metrics(y_true, y_pred, cfg.labels),
        "per_intent_f1": per_intent_f1(y_true, y_pred, cfg.labels),
        "invalid_reasons": rows["invalid_reason"].value_counts().to_dict(),
        "llm": stats,
        "latency": {
            "latency_p50_ms": stats["latency_p50_ms"],
            "latency_p95_ms": stats["latency_p95_ms"],
            "latency_calls": len(rows),
        },
        "hardware": _hardware(),
    }
    if smoke:
        return report

    configure_mlflow(cfg)
    params = {"family": condition, "mode": report["mode"], "seed": cfg.seed, **report["params"]}
    check_no_secrets(params)  # no credential-like params
    with mlflow.start_run(run_name=report["model"]) as run:
        mlflow.set_tags({"family": condition, "mode": report["mode"]})
        mlflow.log_params(params)
        mlflow.log_text(system, "prompt.txt")  # the prompt template
        mlflow.log_metrics({f"test_{k}": v for k, v in report["metrics"].items()})
        mlflow.log_metrics({k: float(v) for k, v in report["llm"].items()})
        report["mlflow_run_id"] = run.info.run_id
        _write_llm_rows(lcfg.results_path, rows)
        out = cfg.evaluation.reports_dir / f"{report['model']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(out), artifact_path="reports")
    return report


_LABELS = {"b0": "B0 majority", "b1": "B1 TF-IDF+LR", "t1": "T1 DistilBERT"}
_LLM_LABELS = {"l1": "L1 zero-shot", "l2": "L2 few-shot"}


def _row_name(report: dict[str, Any]) -> str:
    family = report["family"]
    if family in _LLM_LABELS:
        return f"{_LLM_LABELS[family]} ({report['params']['llm_model']}, local)"
    mode = report["mode"].upper()
    if report.get("sample_size"):
        mode += f" {report['sample_size']}"
    if family == "t1":
        mode += " (CPU)" if report["mode"] == "sample" else " (Colab)"
    return f"{_LABELS[family]} {mode}"


def _eval_set(report: dict[str, Any]) -> str:
    return str(
        report.get("eval_set") or (report["mode"] if report["family"] in _LLM_LABELS else "test")
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "–" if value is None else f"{value:,.{digits}f}"


def build_comparison(cfg: Config) -> str:
    """`reports/model_comparison.md` from `reports/metrics/*.json` only.

    One table per evaluation set, so a 300-message subset row never sits next to a
    full-test row. No number is typed by hand.
    """
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cfg.evaluation.reports_dir.glob("*.json"))
    ]
    order = {"b0": 0, "b1": 1, "t1": 2, "l1": 3, "l2": 4}
    reports.sort(key=lambda r: (order[r["family"]], r["mode"] != "sample", r["model"]))
    header = (
        "| Model | Seeds | Accuracy | Macro-F1 | Weighted-F1 | Category macro-F1 | p50 ms | p95 ms "
        "| Size MB | Avg in tokens | Cost /1k USD | Invalid % | Run id |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        "# Model comparison",
        "",
        "Generated by `make compare` from `reports/metrics/*.json`; do not edit by hand.",
        "Primary metric: intent macro-F1 over 27 intents. Latency: single message, warm, CPU.",
        "",
    ]
    for eval_set in sorted({_eval_set(r) for r in reports}, key=lambda s: (s != "test", s)):
        rows = [r for r in reports if _eval_set(r) == eval_set]
        n = rows[0]["test_rows"]
        lines += [f"## Evaluation set: `{eval_set}` ({n:,} messages)", "", header.rstrip("\n")]
        for r in rows:
            m, lat, llm_stats = r["metrics"], r["latency"], r.get("llm", {})
            size = r.get("model_size_bytes")
            invalid = llm_stats.get("invalid_rate")
            lines.append(
                "| "
                + " | ".join(
                    [
                        _row_name(r),
                        "1",
                        _fmt(m["intent_accuracy"]),
                        _fmt(m["intent_macro_f1"]),
                        _fmt(m["intent_weighted_f1"]),
                        _fmt(m["category_macro_f1"]),
                        _fmt(lat.get("latency_p50_ms"), 3),
                        _fmt(lat.get("latency_p95_ms"), 3),
                        _fmt(None if size is None else size / 1e6, 2),
                        _fmt(llm_stats.get("avg_input_tokens"), 1),
                        _fmt(llm_stats.get("cost_per_1k_usd"), 4),
                        _fmt(None if invalid is None else invalid * 100, 1),
                        str(r.get("mlflow_run_id", "–"))[:8],
                    ]
                )
                + " |"
            )
        lines.append("")
    hardware = {json.dumps(r["hardware"], sort_keys=True) for r in reports if "hardware" in r}
    lines += [
        "## Notes",
        "",
        "- Seeds = 1 for every row (3-seed variance is Advanced and not run yet).",
        "- B0 has no latency row: it is a sanity floor, never served.",
        "- LLM rows: local Ollama model on CPU, $0; not a frontier API model.",
        "- Hardware: " + "; ".join(sorted(hardware)),
        "",
    ]
    return "\n".join(lines)


def write_comparison(cfg: Config) -> Path:
    out = cfg.evaluation.reports_dir.parent / "model_comparison.md"
    out.write_text(build_comparison(cfg), encoding="utf-8")
    return out


def confused_pairs(rows: pd.DataFrame, top: int) -> pd.DataFrame:
    """Most frequent (true intent -> predicted) error pairs, with counts."""
    wrong = rows[rows["intent"] != rows["prediction"]]
    pairs = wrong.groupby(["intent", "prediction"]).size().rename("count").reset_index()
    return pairs.sort_values(["count", "intent"], ascending=[False, True]).head(top)


def plot_errors(rows: pd.DataFrame, labels: LabelConfig, title: str, out: Path) -> None:
    """Off-diagonal confusion matrix: the diagonal is blanked so the few errors are visible.

    Intents are ordered by category, so errors inside one category sit in a block.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    order = sorted(labels.intents, key=lambda i: (labels.intent_to_category[i], i))
    valid = rows[rows["prediction"].isin(order)]
    matrix = confusion_matrix(valid["intent"], valid["prediction"], labels=order).astype(float)
    np.fill_diagonal(matrix, np.nan)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(np.where(matrix > 0, matrix, np.nan), cmap="Blues", vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        if value > 0:
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_xlabel("predicted intent")
    ax.set_ylabel("true intent")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03, label="errors")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def write_error_analysis(cfg: Config, top: int = 8) -> Path:
    """`reports/error_analysis.md` + confusion figures from the saved predictions.

    Example texts come from the hash-checked test split; no model is re-run.
    """
    test = load_split(cfg, "test")
    texts = test["instruction"].tolist()
    sources: dict[str, pd.DataFrame] = {}
    for name in ("b1", "t1"):
        path = cfg.evaluation.predictions_dir / f"{name}.csv"
        if path.exists():
            sources[f"{name.upper()} FULL (test, {len(test):,})"] = pd.read_csv(path)
    if cfg.llm.results_path.exists():
        llm_rows = pd.read_csv(cfg.llm.results_path)
        for cond, rows in llm_rows.groupby("condition"):
            sources[f"{str(cond).upper()} ({cfg.llm.model}, test subset {len(rows)})"] = rows
    figures = cfg.evaluation.reports_dir.parent / "figures"
    lines = [
        "# Error analysis",
        "",
        "Generated by `make errors` from `reports/predictions/*.csv` and "
        "`reports/llm_zero_shot.csv`; example texts from the hash-checked test split.",
        "",
    ]
    for title, rows in sources.items():
        wrong = rows[rows["intent"] != rows["prediction"]]
        same_cat = sum(
            cfg.labels.intent_to_category.get(p) == cfg.labels.intent_to_category[i]
            for i, p in zip(wrong["intent"], wrong["prediction"], strict=True)
        )
        lines += [
            f"## {title}",
            "",
            f"Errors: {len(wrong)} of {len(rows)}; inside the true category: {same_cat}.",
            "",
        ]
        key = title.split()[0].lower()
        if key in ("b1", "t1"):
            figure = figures / f"confusion_{key}.png"
            plot_errors(rows, cfg.labels, f"{title}: errors only (diagonal hidden)", figure)
            lines += [f"![{key} errors](figures/{figure.name})", ""]
        lines += [
            "| True intent | Predicted | Count | Example (test_row: text) |",
            "|---|---|---|---|",
        ]
        for pair in confused_pairs(rows, top).itertuples():
            hits = wrong[
                (wrong["intent"] == pair.intent) & (wrong["prediction"] == pair.prediction)
            ]
            examples = "<br>".join(
                f"{r}: {texts[r]}".replace("|", "\\|") for r in hits["test_row"].head(2)
            )
            lines.append(f"| `{pair.intent}` | `{pair.prediction}` | {pair.count} | {examples} |")
        lines.append("")
    out = cfg.evaluation.reports_dir.parent / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


NOISY_SLICE = 100  # slices with fewer rows are marked as noisy


def slice_metrics(rows: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    """Accuracy and macro-F1 on one slice; macro-F1 over the intents present in the slice,
    so a small slice is not punished for intents it does not contain."""
    part = rows[mask]
    present = sorted(set(part["intent"]))
    m = classification_metrics(part["intent"].tolist(), part["prediction"].tolist(), present)
    return {"rows": len(part), "errors": int((part["intent"] != part["prediction"]).sum()), **m}


def robustness_table(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per `flags` letter (rows containing it) + near-duplicate vs not."""
    letters = sorted({ch for f in rows["flags"] for ch in str(f)})
    slices = {f"flag {ch}": rows["flags"].str.contains(ch, regex=False) for ch in letters}
    near = rows["near_dup_cosine"] >= threshold
    slices[f"near-duplicate in train (cosine >= {threshold:.2f})"] = near
    slices[f"no near-duplicate (cosine < {threshold:.2f})"] = ~near
    slices["all"] = pd.Series(True, index=rows.index)
    return pd.DataFrame([{"slice": k, **slice_metrics(rows, v)} for k, v in slices.items()])


def write_robustness(cfg: Config) -> Path:
    """`reports/robustness.md` from the saved predictions."""
    tables = {}
    for name in ("b1", "t1"):
        path = cfg.evaluation.predictions_dir / f"{name}.csv"
        tables[name] = robustness_table(pd.read_csv(path), cfg.split.near_duplicate_threshold)
    merged = tables["b1"].merge(tables["t1"], on=["slice", "rows"], suffixes=("_b1", "_t1"))
    lines = [
        "# Robustness",
        "",
        "Generated by `make robustness` from `reports/predictions/{b1,t1}.csv` (FULL models, "
        "full test split). A row counts in a `flags` slice when its flags contain the letter.",
        f"Slices under {NOISY_SLICE} rows are marked noisy: one error moves accuracy by more "
        "than 1 point. Macro-F1 is over the intents present in the slice.",
        "",
        "## Per `flags` letter and near-duplicate slice",
        "",
        "| Slice | Rows | B1 errors | B1 accuracy | B1 macro-F1 | T1 errors | T1 accuracy "
        "| T1 macro-F1 | Noisy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in merged.to_dict("records"):
        lines.append(
            f"| {r['slice']} | {r['rows']:,} | {r['errors_b1']} | {r['accuracy_b1']:.4f} | "
            f"{r['macro_f1_b1']:.4f} | {r['errors_t1']} | {r['accuracy_t1']:.4f} | "
            f"{r['macro_f1_t1']:.4f} | {'yes' if r['rows'] < NOISY_SLICE else ''} |"
        )
    lines += [
        "",
        "Flag meanings: see the Bitext dataset card (`S` and `V` undocumented). "
        "`Z` = errors and typos.",
        "",
        "## Typo perturbation",
        "",
    ]
    typos_path = cfg.evaluation.reports_dir.parent / "robustness_typos.json"
    if not typos_path.exists():
        lines += ["Not run yet: `make typos`.", ""]
    else:
        typos = json.loads(typos_path.read_text(encoding="utf-8"))
        rates = ["0.0"] + [str(r) for r in typos["rates"]]
        lines += [
            f"Seeded (seed {typos['seed']}) character noise on all {typos['test_rows']:,} test "
            f"rows: each word of >= {typos['min_word_len']} letters gets one edit (swap, delete, "
            "insert, QWERTY neighbour) with probability `rate`. Source: "
            "`reports/robustness_typos.json` (`make typos`).",
            "",
            "| Model | "
            + " | ".join(f"acc @ {r}" for r in rates)
            + " | "
            + " | ".join(f"drop @ {r} (points)" for r in rates[1:])
            + " |",
            "|---|" + "---|" * (2 * len(rates) - 1),
        ]
        for family, entry in typos["models"].items():
            acc = [entry["scores"][r]["accuracy"] for r in rates]
            drops = [(acc[0] - a) * 100 for a in acc[1:]]
            lines.append(
                f"| {family.upper()} FULL | "
                + " | ".join(f"{a:.4f}" for a in acc)
                + " | "
                + " | ".join(f"{d:.2f}" for d in drops)
                + " |"
            )
        lines += ["", "Examples (first test rows):", ""]
        for key in rates:
            shown = " / ".join(f"`{x}`" for x in typos["examples"][key][:2])
            lines += [f"- rate {key}: {shown}"]
        lines.append("")
    out = cfg.evaluation.reports_dir.parent / "robustness.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


_KEY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_TOKEN = re.compile(r"(\{\{[^}]*\}\}|[A-Za-z]+)")  # placeholders first, so they stay whole


def _keyboard_neighbours() -> dict[str, str]:
    """Letters next to each letter on a QWERTY keyboard (same row and the rows above/below)."""
    near: dict[str, str] = {}
    for r, row in enumerate(_KEY_ROWS):
        for c, ch in enumerate(row):
            keys = {row[i] for i in (c - 1, c + 1) if 0 <= i < len(row)}
            for other in (r - 1, r + 1):
                if 0 <= other < len(_KEY_ROWS):
                    up_down = _KEY_ROWS[other]
                    keys |= {up_down[i] for i in (c - 1, c) if 0 <= i < len(up_down)}
            near[ch] = "".join(sorted(keys))
    return near


_NEAR = _keyboard_neighbours()


def _typo(word: str, rng: random.Random) -> str:
    """One edit: swap two adjacent letters, delete, insert a random letter, or QWERTY neighbour."""
    op = rng.choice(("swap", "delete", "insert", "replace"))
    i = rng.randrange(len(word))
    if op == "swap":
        i = min(i, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if op == "delete":
        return word[:i] + word[i + 1 :]
    if op == "insert":
        return word[:i] + rng.choice(string.ascii_lowercase) + word[i:]
    new = rng.choice(_NEAR[word[i].lower()])
    return word[:i] + (new.upper() if word[i].isupper() else new) + word[i + 1 :]


def perturb_text(text: str, rate: float, rng: random.Random, min_len: int) -> str:
    """Each alphabetic word of >= `min_len` letters gets one typo with probability `rate`.
    `{{Placeholders}}` and non-letters are untouched. The caller owns the seeded `rng`."""
    parts = _TOKEN.split(text)
    for i, part in enumerate(parts):
        if part.isalpha() and len(part) >= min_len and rng.random() < rate:
            parts[i] = _typo(part, rng)
    return "".join(parts)


def evaluate_typo_robustness(cfg: Config) -> Path:
    """Accuracy/macro-F1 of B1 FULL and T1 FULL on clean vs typo-perturbed test."""
    test = load_split(cfg, "test")  # hash-checked
    texts, y_true = test["instruction"].tolist(), test["intent"].tolist()
    noisy: dict[float, list[str]] = {}
    min_len = cfg.evaluation.typo_min_word_len
    for rate in cfg.evaluation.typo_rates:
        rng = random.Random(cfg.seed)  # one fresh seeded stream per rate, rows in test order
        noisy[rate] = [perturb_text(x, rate, rng, min_len) for x in texts]
    result: dict[str, Any] = {
        "seed": cfg.seed,
        "rates": list(cfg.evaluation.typo_rates),
        "min_word_len": cfg.evaluation.typo_min_word_len,
        "test_rows": len(test),
        "examples": {str(rate): noisy[rate][:3] for rate in noisy} | {"0.0": texts[:3]},
        "models": {},
    }
    for family in ("b1", "t1"):
        run = find_run(cfg, family, None)
        model = load_checked_model(cfg, run)
        clean_pred = model.predict(texts)[0]
        scores = {"0.0": classification_metrics(y_true, clean_pred, cfg.labels.intents)}
        for rate, perturbed in noisy.items():
            pred = model.predict(perturbed)[0]
            scores[str(rate)] = classification_metrics(y_true, pred, cfg.labels.intents)
        result["models"][family] = {"mlflow_run_id": run.info.run_id, "scores": scores}
        with mlflow.start_run(run_id=run.info.run_id):
            for key, m in scores.items():
                mlflow.log_metric(f"typo{key}_test_accuracy", m["accuracy"])
    out = cfg.evaluation.reports_dir.parent / "robustness_typos.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out


REVIEW_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def coverage_curve(
    y_true: Sequence[str], y_pred: Sequence[str], confidence: Sequence[float]
) -> list[dict[str, float]]:
    """Threshold curve: for each threshold t, messages with confidence >= t are routed
    automatically (coverage, accuracy, errors that slip through); the rest go to a human."""
    correct = np.asarray(y_true) == np.asarray(y_pred)
    conf = np.asarray(confidence)
    rows = []
    for t in REVIEW_THRESHOLDS:
        auto = conf >= t
        rows.append(
            {
                "threshold": t,
                "coverage": float(auto.mean()),
                "auto_accuracy": float(correct[auto].mean()) if auto.any() else 1.0,
                "auto_errors": int((auto & ~correct).sum()),
                "to_review": int((~auto).sum()),
                "errors_caught": int((~auto & ~correct).sum()),
            }
        )
    return rows


def write_threshold_curve(cfg: Config) -> Path:
    """Coverage/accuracy per threshold on **validation** for @champion and @fallback.

    The threshold is chosen on val, never on test."""
    from supportrouter.predict import Predictor

    val = load_split(cfg, "val")
    texts, y_true = val["instruction"].tolist(), val["intent"].tolist()
    result: dict[str, Any] = {"split": "val", "rows": len(val), "models": {}}
    lines = [
        "# Review threshold curve (validation)",
        "",
        "Generated by `make threshold-curve`. A message is routed automatically "
        "when confidence >= threshold, otherwise `needs_human_review = true`.",
        "",
    ]
    for alias in ("champion", "fallback"):
        predictor = Predictor.from_registry(cfg, alias)
        preds = predictor.predict(texts)
        curve = coverage_curve(y_true, [p.intent for p in preds], [p.confidence for p in preds])
        result["models"][alias] = {"model_version": predictor.model_version, "curve": curve}
        lines += [
            f"## @{alias}: {predictor.model_version} ({len(val):,} val messages)",
            "",
            "| Threshold | Coverage | Auto accuracy | Auto errors | To review | Errors caught |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['threshold']:.2f} | {r['coverage']:.4f} | {r['auto_accuracy']:.4f} | "
            f"{r['auto_errors']} | {r['to_review']} | {r['errors_caught']} |"
            for r in curve
        ]
        lines.append("")
    out_dir = cfg.evaluation.reports_dir.parent
    (out_dir / "threshold_curve.json").write_text(json.dumps(result, indent=2) + "\n")
    out = out_dir / "threshold_curve.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def measure_api_latency(cfg: Config, base_url: str) -> Path:
    """End-to-end `POST /classify` latency of a running service (API target).

    Single message per request, warm (warm-up calls discarded), over real HTTP on localhost,
    texts from the hash-checked test split. Uses the standard library only (no dev deps).
    """
    import urllib.request

    def call(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - localhost URL
            body: dict[str, Any] = json.loads(resp.read())
            return body

    health = call("/health")
    texts = load_split(cfg, "test")["instruction"].tolist()

    def predict(batch: Sequence[str]) -> tuple[list[str], list[float]]:
        results = call("/classify", {"texts": list(batch)})["results"]
        return [r["intent"] for r in results], [r["confidence"] for r in results]

    report = {
        "endpoint": "POST /classify",
        "model_version": health["model_version"],
        "latency": measure_latency(predict, texts, cfg),
        "target_p95_ms": 250.0 if "(t1," in health["model_version"] else 100.0,
        "hardware": _hardware(),
    }
    out = cfg.evaluation.reports_dir.parent / "api_latency.json"  # not a model metrics file
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["b0", "b1", "t1", "l1", "l2"], required=True)
    parser.add_argument("--predictions", action="store_true", help="write per-row predictions")
    parser.add_argument("--api-url", default=None, help="measure latency of a running API")
    parser.add_argument("--threshold-curve", action="store_true", help="threshold curve on val")
    parser.add_argument("--typos", action="store_true", help="B1/T1 FULL typo robustness")
    parser.add_argument("--subset", action="store_true", help="b0/b1/t1 on the L1/L2 subset")
    parser.add_argument(
        "--smoke", type=int, default=None, help="LLM only: N val messages, no files"
    )
    parser.add_argument("--sample", type=int, nargs="?", const=cfg.sample_size, default=None)
    args = parser.parse_args()
    if args.api_url:
        print(measure_api_latency(cfg, args.api_url.rstrip("/")))
        return
    if args.threshold_curve:
        print(write_threshold_curve(cfg))
        return
    if args.typos:
        print(evaluate_typo_robustness(cfg))
        return
    if args.predictions:
        print(write_test_predictions(cfg, args.model, sample=args.sample))
        return
    if args.model in ("l1", "l2"):
        report = evaluate_llm(cfg, args.model, smoke=args.smoke)
        extra = report["llm"]
    else:
        report = evaluate_on_test(cfg, args.model, sample=args.sample, subset=args.subset)
        extra = report["latency"]
    keys = ("model", "mlflow_run_id", "mode", "test_rows")
    summary = {k: report[k] for k in keys if k in report}
    print(json.dumps({**summary, **report["metrics"], **extra}, indent=2))


if __name__ == "__main__":
    main()
