"""CI quality gate (Phase 10).

Trains B1 (TF-IDF + LogisticRegression) on the stratified SAMPLE of train with the
normal training code, scores it on the full validation split and fails (exit code 1) when
intent macro-F1 is below `quality_gate.min_val_macro_f1` (0.85). The test split is never
read. Run: `python -m supportrouter.quality_gate [--sample N]` or `make quality-gate`.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

from supportrouter.config import Config, load_config
from supportrouter.train import train_baseline


@dataclass(frozen=True)
class GateResult:
    model: str
    mode: str
    sample_size: int | None
    split: str
    run_id: str
    val_intent_macro_f1: float
    threshold: float
    threshold_source: str
    passed: bool


def gate_threshold(cfg: Config) -> tuple[float, str]:
    """Threshold from config, or from the override env var (used only to prove failure)."""
    raw = os.environ.get(cfg.quality_gate.threshold_env)
    if raw is None or raw == "":
        return cfg.quality_gate.min_val_macro_f1, "config"
    value = float(raw)
    if not 0 < value <= 1:
        raise ValueError(f"{cfg.quality_gate.threshold_env} must be in (0, 1], got {raw!r}")
    return value, "env"


def run_gate(cfg: Config, sample: int | None) -> GateResult:
    """Train B1 SAMPLE through `train_baseline` (MLflow run, checksum) and compare on val."""
    threshold, source = gate_threshold(cfg)
    result = train_baseline(cfg, "b1", sample)
    score = result.val_metrics["intent_macro_f1"]
    gate = GateResult(
        model="b1",
        mode=str(result.params["mode"]),
        sample_size=sample,
        split="val",
        run_id=result.run_id,
        val_intent_macro_f1=score,
        threshold=threshold,
        threshold_source=source,
        passed=score >= threshold,
    )
    path = cfg.quality_gate.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(gate), indent=2) + "\n", encoding="utf-8")
    return gate


def main(argv: list[str] | None = None, cfg: Config | None = None) -> int:
    """Exit code 0 = pass, 1 = below threshold (fails the CI job)."""
    cfg = cfg or load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=cfg.sample_size)
    args = parser.parse_args(argv)

    gate = run_gate(cfg, args.sample)
    verdict = "PASS" if gate.passed else "FAIL"
    print(
        f"quality gate {verdict}: B1 SAMPLE{gate.sample_size} val intent macro-F1 "
        f"{gate.val_intent_macro_f1:.4f} vs threshold {gate.threshold} ({gate.threshold_source}); "
        f"run {gate.run_id[:8]} -> {cfg.quality_gate.report_path.name}"
    )
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
