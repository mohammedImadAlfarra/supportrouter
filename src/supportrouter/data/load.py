"""Raw data acquisition: download the pinned Bitext dataset and describe it."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from supportrouter.config import Config, DataConfig
from supportrouter.data.validate import validate_raw

TARGET_COLUMNS = ("intent", "category")
MANIFEST_NAME = "split_manifest.json"


class SplitHashError(RuntimeError):
    """A split file does not match the SHA-256 recorded in the split manifest."""


def download_raw(cfg: DataConfig, *, force: bool = False) -> Path:
    """Download the dataset at the pinned revision and write it to `cfg.raw_path` as CSV.

    An existing file is kept unless `force` is set, so re-running never silently replaces data.
    """
    if cfg.raw_path.exists() and not force:
        return cfg.raw_path
    dataset = load_dataset(cfg.hf_dataset_id, revision=cfg.hf_revision, split="train")
    frame: pd.DataFrame = dataset.to_pandas()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.raw_path, index=False)
    return cfg.raw_path


def read_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV with every column as text and no NA guessing.

    pandas' default turns strings such as "NA" or "None" into missing values;
    the first look must show the file as it is.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_validated(cfg: Config) -> pd.DataFrame:
    """Read the raw CSV and validate it against the schema before any cleaning."""
    return validate_raw(read_raw(cfg.data.raw_path), cfg.labels)


def load_split(cfg: Config, name: str) -> pd.DataFrame:
    """Read one frozen split after checking its hash against the manifest.

    The frame is also validated with the raw schema, so a hand-edited file
    fails twice: on its hash and on any bad label.
    """
    split_dir = cfg.split.processed_dir
    manifest = json.loads((split_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    path = split_dir / f"{name}.csv"
    expected = manifest["splits"][name]["sha256"]
    actual = file_sha256(path)
    if actual != expected:
        raise SplitHashError(f"{path.name}: sha256 {actual[:12]}… != manifest {expected[:12]}…")
    return validate_raw(read_raw(path), cfg.labels)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_raw(frame: pd.DataFrame) -> dict[str, Any]:
    """First-look facts for the data card: shape, empties, label and `flags` counts."""
    summary: dict[str, Any] = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "empty_values": {col: int((frame[col].str.strip() == "").sum()) for col in frame.columns},
    }
    for col in (*TARGET_COLUMNS, "flags"):
        if col in frame.columns:
            counts = frame[col].value_counts()
            summary[f"{col}_n_unique"] = int(counts.size)
            summary[f"{col}_counts"] = {str(k): int(v) for k, v in counts.items()}
    return summary
