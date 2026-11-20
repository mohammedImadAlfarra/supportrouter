import json
from pathlib import Path

import pandas as pd
import pytest

from supportrouter import quality_gate as gate_module
from supportrouter import train as train_module
from supportrouter.config import Config
from supportrouter.quality_gate import gate_threshold, main, run_gate
from supportrouter.train import TrainResult


def fake_training(score: float) -> object:
    def fake(cfg: Config, family: str, sample: int | None) -> TrainResult:
        assert family == "b1"
        params = {"mode": "sample" if sample else "full"}
        return TrainResult("f" * 32, Path("unused"), params, {"intent_macro_f1": score})

    return fake


def test_raised_threshold_fails_the_gate(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    # Same score, two thresholds: only the raised one must give a non-zero exit code.
    monkeypatch.setattr(gate_module, "train_baseline", fake_training(0.90))
    monkeypatch.delenv(cfg.quality_gate.threshold_env, raising=False)
    assert main(["--sample", "30"], cfg) == 0
    monkeypatch.setenv(cfg.quality_gate.threshold_env, "0.95")
    assert main(["--sample", "30"], cfg) == 1
    report = json.loads(cfg.quality_gate.report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["threshold"] == 0.95
    assert report["threshold_source"] == "env"


def test_real_training_below_threshold_fails_and_reads_val_only(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real training code path on the tiny fixture split, whose toy texts B1 cannot learn
    # (val macro-F1 far below 0.85): the gate must fail with exit code 1, and test.csv is
    # never read.
    monkeypatch.delenv(cfg.quality_gate.threshold_env, raising=False)
    seen: list[str] = []
    real = train_module.load_split

    def spy(c: Config, name: str) -> pd.DataFrame:
        seen.append(name)
        return real(c, name)

    monkeypatch.setattr(train_module, "load_split", spy)
    gate = run_gate(cfg, sample=30)
    assert sorted(set(seen)) == ["train", "val"]
    assert (gate.model, gate.mode, gate.sample_size, gate.split) == ("b1", "sample", 30, "val")
    assert gate.threshold == cfg.quality_gate.min_val_macro_f1 == 0.85
    assert gate.val_intent_macro_f1 < 0.85
    assert not gate.passed
    saved = json.loads(cfg.quality_gate.report_path.read_text(encoding="utf-8"))
    assert saved["val_intent_macro_f1"] == gate.val_intent_macro_f1
    assert saved["run_id"] == gate.run_id
    assert main(["--sample", "30"], cfg) == 1


@pytest.mark.parametrize("raw", ["0", "1.5", "-0.1"])
def test_threshold_override_must_be_a_probability(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(cfg.quality_gate.threshold_env, raw)
    with pytest.raises(ValueError, match="must be in"):
        gate_threshold(cfg)


def test_empty_override_falls_back_to_config(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cfg.quality_gate.threshold_env, "")
    assert gate_threshold(cfg) == (0.85, "config")
