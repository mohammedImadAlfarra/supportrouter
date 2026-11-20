"""Evidently drift report: logged traffic vs the reference batch (Phase 11).

Reads only the prediction log (no raw text) and the request_ids written by
`scripts/simulate_traffic.py`. Columns: `intent`, `category` (categorical) and
`confidence`, `text_chars`, `text_words` (numerical). For each current batch (`normal`,
`shifted`) Evidently's `DataDriftPreset` picks the test per column (by type and reference size),
and its test statuses decide drift; this module does not re-implement the thresholds.
Outputs: one HTML per batch (git-ignored, ~3.7 MB) and a small `summary.json` / `summary.md`.
Acceptance: `shifted` is flagged and `normal` is not; otherwise exit code 1.
Run: `make drift` (sets `DO_NOT_TRACK=1`).
"""

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from supportrouter.config import Config, load_config

CATEGORICAL = ["intent", "category"]
NUMERICAL = ["confidence", "text_chars", "text_words"]
CURRENT_BATCHES = ("normal", "shifted")


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    method: str
    threshold: float
    score: float
    drifted: bool


@dataclass(frozen=True)
class BatchDrift:
    batch: str
    rows: int
    drifted_columns: int
    drift_share: float
    dataset_drift: bool
    columns: list[ColumnDrift]


def load_batch_frames(cfg: Config) -> tuple[dict[str, pd.DataFrame], str]:
    """Log lines of each simulated batch, matched by request_id; returns frames + model version."""
    meta = json.loads(cfg.monitoring.batches_path.read_text(encoding="utf-8"))
    owner = {rid: name for name, b in meta["batches"].items() for rid in b["request_ids"]}
    to_category = cfg.labels.intent_to_category
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in meta["batches"]}
    with cfg.api.log_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            name = owner.get(rec["request_id"])
            if name is None:
                continue  # other traffic in the log is not part of this experiment
            if "text_chars" not in rec:
                raise ValueError("log line without text length fields; re-run simulate-traffic")
            rows[name].append(
                {
                    "intent": rec["intent"],
                    "category": to_category[rec["intent"]],
                    "confidence": float(rec["confidence"]),
                    "text_chars": int(rec["text_chars"]),
                    "text_words": int(rec["text_words"]),
                }
            )
    frames = {name: pd.DataFrame(r, columns=CATEGORICAL + NUMERICAL) for name, r in rows.items()}
    for name, b in meta["batches"].items():
        if len(frames[name]) != b["messages"]:
            raise ValueError(f"{name}: {len(frames[name])} log lines for {b['messages']} messages")
    return frames, str(meta["model_version"])


def run_drift(
    reference: pd.DataFrame, current: pd.DataFrame, drift_share: float, batch: str
) -> tuple[Any, BatchDrift]:
    """Run Evidently on one batch; return the snapshot (for HTML) and the parsed result."""
    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    report = Report([DataDriftPreset(drift_share=drift_share)], include_tests=True)
    snapshot = report.run(  # Evidently order: current first, then reference
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    result = snapshot.dict()
    scores = {m["id"]: m["value"] for m in result["metrics"]}
    columns, share, dataset_drift = [], 0.0, False
    for test in result["tests"]:
        params = test["metric_config"]["params"]
        failed = test["status"] == "FAIL"
        if params["type"].endswith(":ValueDrift"):
            columns.append(
                ColumnDrift(
                    column=params["column"],
                    method=params["method"],
                    threshold=float(params["threshold"]),
                    score=float(scores[test["metric_config"]["metric_id"]]),
                    drifted=failed,
                )
            )
        elif params["type"].endswith(":DriftedColumnsCount"):
            share = float(scores[test["metric_config"]["metric_id"]]["share"])
            dataset_drift = failed  # Evidently: drift when share >= drift_share
    if {c.column for c in columns} != set(CATEGORICAL + NUMERICAL):
        raise ValueError(f"unexpected Evidently tests: {[c.column for c in columns]}")
    drifted = sum(c.drifted for c in columns)
    return snapshot, BatchDrift(batch, len(current), drifted, share, dataset_drift, columns)


def _profile(frame: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Plain numbers next to the tests, so the summary can be read without Evidently."""
    return {
        "rows": len(frame),
        "share_in_shift_categories": round(
            float(frame["category"].isin(cfg.monitoring.shift_categories).mean()), 4
        ),
        "mean_confidence": round(float(frame["confidence"].mean()), 4),
        "share_needs_review": round(
            float((frame["confidence"] < cfg.api.review_threshold).mean()), 4
        ),
        "mean_text_chars": round(float(frame["text_chars"].mean()), 2),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Drift report summary (Phase 11)",
        "",
        f"Generated by `make drift` on {summary['created']}; model `{summary['model_version']}`.",
        "Reference = half of the validation split sent through the API; "
        "columns hold no raw text. Evidently "
        f"{summary['evidently']}, `DataDriftPreset(drift_share={summary['drift_share']})`.",
        "",
        f"| batch | rows | in {'/'.join(summary['shift_categories'])} | mean confidence "
        "| needs review | mean chars |",
        "|---|---|---|---|---|---|",
    ]
    for name, p in summary["profiles"].items():
        lines.append(
            f"| {name} | {p['rows']} | {p['share_in_shift_categories']:.4f} | "
            f"{p['mean_confidence']:.4f} | {p['share_needs_review']:.4f} | "
            f"{p['mean_text_chars']:.2f} |"
        )
    for b in summary["batches"]:
        lines += [
            "",
            f"## `{b['batch']}` vs `reference`: dataset drift = **{b['dataset_drift']}** "
            f"({b['drifted_columns']}/{len(b['columns'])} columns, share {b['drift_share']:.2f})",
            "",
            "| column | method | threshold | score | drifted |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {c['column']} | {c['method']} | {c['threshold']} | {c['score']:.4g} | "
            f"{c['drifted']} |"
            for c in b["columns"]
        ]
    verdict = "PASS" if summary["acceptance_passed"] else "FAIL"
    lines += [
        "",
        f"**Acceptance: {verdict}** — shifted flagged, normal not flagged.",
        "",
        "HTML reports (git-ignored): " + ", ".join(f"`{p}`" for p in summary["html"]),
        "",
    ]
    return "\n".join(lines)


def write_drift_reports(cfg: Config) -> tuple[Path, bool]:
    """Write the HTML reports and the summary; return the summary path and the acceptance."""
    import evidently

    frames, model_version = load_batch_frames(cfg)
    out_dir = cfg.monitoring.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results, html = [], []
    for name in CURRENT_BATCHES:
        snapshot, res = run_drift(
            frames["reference"], frames[name], cfg.monitoring.drift_share, name
        )
        path = out_dir / f"drift_{name}.html"
        snapshot.save_html(str(path))
        html.append(path.name)
        results.append(res)
    by_name = {r.batch: r for r in results}
    passed = by_name["shifted"].dataset_drift and not by_name["normal"].dataset_drift
    summary = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": model_version,
        "evidently": evidently.__version__,
        "drift_share": cfg.monitoring.drift_share,
        "shift_categories": list(cfg.monitoring.shift_categories),
        "profiles": {name: _profile(f, cfg) for name, f in frames.items()},
        "batches": [asdict(r) for r in results],
        "acceptance_passed": passed,
        "html": html,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = out_dir / "summary.md"
    md.write_text(_markdown(summary), encoding="utf-8")
    return md, passed


def main() -> None:
    path, passed = write_drift_reports(load_config())
    print(path)
    print("acceptance:", "PASS (shifted flagged, normal not)" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
