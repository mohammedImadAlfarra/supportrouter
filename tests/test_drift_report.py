"""Phase 11: drift report on a synthetic prediction log (no server, no raw text)."""

import dataclasses
import json
import random
from pathlib import Path

import pytest

from supportrouter.config import Config
from supportrouter.monitoring.drift_report import load_batch_frames, write_drift_reports

SIZES = {"reference": 1200, "normal": 300, "shifted": 300}


def write_traffic(cfg: Config, tmp: Path, length_fields: bool = True) -> Config:
    """Log lines in the PredictionLog format; `shifted` has a skewed intent mix and lower
    confidence, `normal` is drawn from the same distribution as `reference`."""
    cfg = dataclasses.replace(
        cfg,
        api=dataclasses.replace(cfg.api, log_path=tmp / "predictions.jsonl"),
        monitoring=dataclasses.replace(
            cfg.monitoring, batches_path=tmp / "batches.json", report_dir=tmp / "drift"
        ),
    )
    rng = random.Random(42)
    intents = list(cfg.labels.intents)
    lines, batches = [], {}
    for name, n in SIZES.items():
        ids = []
        for start in range(0, n, 32):
            rid = f"{name}-{start}"
            ids.append(rid)
            for item in range(min(32, n - start)):
                shifted = name == "shifted"
                intent = "cancel_order" if shifted and rng.random() < 0.7 else rng.choice(intents)
                words = rng.randint(8, 12)
                rec = {
                    "request_id": rid,
                    "item": item,
                    "text_hash": "0" * 64,
                    "intent": intent,
                    "confidence": rng.uniform(0.6, 0.95) if shifted else rng.uniform(0.98, 1.0),
                }
                if length_fields:
                    rec |= {"text_chars": words * 5, "text_words": words}
                lines.append(json.dumps(rec))
        batches[name] = {"messages": n, "request_ids": ids}
    lines.append(json.dumps({"request_id": "other-traffic", "intent": "review"}))
    cfg.api.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"model_version": "test/1", "batches": batches}
    cfg.monitoring.batches_path.write_text(json.dumps(meta), encoding="utf-8")
    return cfg


def test_shifted_batch_is_flagged_and_normal_is_not(cfg: Config, tmp_path: Path) -> None:
    cfg = write_traffic(cfg, tmp_path)
    md, passed = write_drift_reports(cfg)
    summary = json.loads((md.parent / "summary.json").read_text())
    assert passed and summary["acceptance_passed"]
    by_batch = {b["batch"]: b for b in summary["batches"]}
    assert by_batch["shifted"]["dataset_drift"] and not by_batch["normal"]["dataset_drift"]
    shifted_cols = {c["column"]: c["drifted"] for c in by_batch["shifted"]["columns"]}
    assert shifted_cols["confidence"] and shifted_cols["intent"]
    assert summary["profiles"]["reference"]["rows"] == 1200  # other traffic is ignored
    assert (md.parent / "drift_shifted.html").stat().st_size > 0
    assert "PASS" in md.read_text()


def test_log_without_length_fields_is_refused(cfg: Config, tmp_path: Path) -> None:
    cfg = write_traffic(cfg, tmp_path, length_fields=False)
    with pytest.raises(ValueError, match="length fields"):
        load_batch_frames(cfg)


def test_missing_log_lines_are_refused(cfg: Config, tmp_path: Path) -> None:
    cfg = write_traffic(cfg, tmp_path)
    meta = json.loads(cfg.monitoring.batches_path.read_text())
    meta["batches"]["normal"]["messages"] = 301
    cfg.monitoring.batches_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="301 messages"):
        load_batch_frames(cfg)
