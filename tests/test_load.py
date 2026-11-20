from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from supportrouter.config import DataConfig
from supportrouter.data import load

ROWS = {
    "flags": ["B", "BQZ", "B"],
    "instruction": ["cancel my order", "where is my refund", "NA"],
    "category": ["ORDER", "REFUND", "ORDER"],
    "intent": ["cancel_order", "track_refund", "cancel_order"],
    "response": ["Sure ...", "Let me check ...", "Sure ..."],
}
REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"


@pytest.fixture
def cfg(tmp_path: Path) -> DataConfig:
    return DataConfig("org/name", REVISION, tmp_path / "raw", "raw.csv")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the network call with an in-memory Dataset and record its arguments."""
    seen: list[dict[str, Any]] = []

    def fake_load_dataset(path: str, **kwargs: Any) -> Dataset:
        seen.append({"path": path, **kwargs})
        return Dataset.from_dict(ROWS)

    monkeypatch.setattr(load, "load_dataset", fake_load_dataset)
    return seen


def test_download_uses_pinned_revision_and_writes_csv(
    cfg: DataConfig, calls: list[dict[str, Any]]
) -> None:
    path = load.download_raw(cfg)
    assert calls == [{"path": "org/name", "revision": REVISION, "split": "train"}]
    assert list(load.read_raw(path).columns) == list(ROWS)


def test_existing_file_is_not_replaced_without_force(
    cfg: DataConfig, calls: list[dict[str, Any]]
) -> None:
    load.download_raw(cfg)
    load.download_raw(cfg)
    assert len(calls) == 1
    load.download_raw(cfg, force=True)
    assert len(calls) == 2


def test_read_raw_keeps_na_like_text(cfg: DataConfig, calls: list[dict[str, Any]]) -> None:
    frame = load.read_raw(load.download_raw(cfg))
    assert frame.loc[2, "instruction"] == "NA"


def test_describe_counts_labels_and_flags(cfg: DataConfig, calls: list[dict[str, Any]]) -> None:
    summary = load.describe_raw(load.read_raw(load.download_raw(cfg)))
    assert summary["rows"] == 3
    assert summary["intent_n_unique"] == 2
    assert summary["intent_counts"] == {"cancel_order": 2, "track_refund": 1}
    assert summary["category_counts"] == {"ORDER": 2, "REFUND": 1}
    assert summary["flags_counts"] == {"B": 2, "BQZ": 1}
    assert summary["empty_values"]["instruction"] == 0


def test_file_sha256_is_stable(cfg: DataConfig, calls: list[dict[str, Any]]) -> None:
    path = load.download_raw(cfg)
    assert load.file_sha256(path) == load.file_sha256(path)
    assert len(load.file_sha256(path)) == 64
