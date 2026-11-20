"""Frozen train/val/test split and exact-leakage check.

The split runs on cleaned, deduplicated data, so every normalized text is already unique;
`check_exact_leakage` proves it on the output instead of trusting the order of steps.
"""

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from supportrouter.config import Config, SplitConfig
from supportrouter.data.clean import clean, normalize
from supportrouter.data.load import MANIFEST_NAME, file_sha256, load_validated

SPLIT_NAMES = ("train", "val", "test")
NEAR_DUPLICATE_RULE = "TF-IDF char 3-5-gram cosine to nearest train row, normalized text"
_CHUNK_ROWS = 1000


class LeakageError(ValueError):
    """The same normalized instruction appears in two splits."""


class FrozenSplitError(RuntimeError):
    """The frozen split on disk differs from what this run would write."""


def sample_stratified(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """SAMPLE mode: an n-row subset with the same `intent` proportions."""
    subset: pd.DataFrame
    subset, _ = train_test_split(frame, train_size=n, stratify=frame["intent"], random_state=seed)
    return subset.reset_index(drop=True)


def stratified_split(frame: pd.DataFrame, cfg: SplitConfig, seed: int) -> dict[str, pd.DataFrame]:
    """Two-step stratified split by `intent`: train vs rest, then rest into val and test.

    Same input + same seed gives the same rows in the same order.
    """
    train, rest = train_test_split(
        frame, train_size=cfg.train, stratify=frame["intent"], random_state=seed
    )
    val, test = train_test_split(
        rest, train_size=cfg.val / (cfg.val + cfg.test), stratify=rest["intent"], random_state=seed
    )
    parts = {"train": train, "val": val, "test": test}
    return {name: parts[name].reset_index(drop=True) for name in SPLIT_NAMES}


def check_exact_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any normalized instruction is shared by two splits.

    The message carries counts only, never customer text.
    """
    keys = {name: set(part["instruction"].map(normalize)) for name, part in splits.items()}
    for a, b in combinations(keys, 2):
        shared = keys[a] & keys[b]
        if shared:
            raise LeakageError(f"{len(shared)} normalized instructions appear in both {a} and {b}")


def nearest_train_cosine(train_texts: pd.Series, texts: pd.Series) -> np.ndarray:
    """Per row of `texts`: cosine to its nearest train row (char 3-5-gram TF-IDF).

    The vectorizer is fitted on train only. TF-IDF rows are L2-normalized, so the dot
    product is the cosine; rows are compared in chunks to keep memory bounded.
    """
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), preprocessor=normalize)
    train_vecs = vectorizer.fit_transform(train_texts)
    vecs = vectorizer.transform(texts)
    best = [
        (vecs[i : i + _CHUNK_ROWS] @ train_vecs.T).max(axis=1).toarray().ravel()
        for i in range(0, vecs.shape[0], _CHUNK_ROWS)
    ]
    return np.concatenate(best)


def near_duplicate_counts(splits: dict[str, pd.DataFrame], threshold: float) -> dict[str, int]:
    """Count val/test rows whose nearest train row has cosine >= threshold (report only)."""
    train = splits["train"]["instruction"]
    return {
        name: int((nearest_train_cosine(train, splits[name]["instruction"]) >= threshold).sum())
        for name in ("val", "test")
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Exact bytes written to disk, so the manifest hash equals the file hash."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _split_entry(part: pd.DataFrame, payload: bytes) -> dict[str, Any]:
    counts = part["intent"].value_counts().sort_index()
    return {
        "rows": len(part),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "intent_counts": {str(k): int(v) for k, v in counts.items()},
        "intent_share": {str(k): round(int(v) / len(part), 6) for k, v in counts.items()},
    }


def write_frozen_splits(
    splits: dict[str, pd.DataFrame], out_dir: Path, header: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Write `{train,val,test}.csv` and the manifest once.

    If a manifest already exists, the run must reproduce it exactly, otherwise
    `FrozenSplitError`. Existing files must still match their hashes; a missing file is
    restored from this identical run (CI rebuilds the split from the pinned raw data
    and the committed manifest). Returns the manifest and whether any file was written.
    """
    payloads = {name: _csv_bytes(part) for name, part in splits.items()}
    manifest = {
        **header,
        "splits": {name: _split_entry(splits[name], payloads[name]) for name in SPLIT_NAMES},
    }
    text = json.dumps(manifest, indent=2) + "\n"
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != text:
            raise FrozenSplitError(
                f"{manifest_path} differs from this run; regenerating a frozen split "
                "is refused (rebuild it deliberately)"
            )
        restored = False
        for name in SPLIT_NAMES:
            path = out_dir / f"{name}.csv"
            if not path.exists():  # same manifest text => these bytes have the frozen hash
                path.write_bytes(payloads[name])
                restored = True
            elif file_sha256(path) != manifest["splits"][name]["sha256"]:
                raise FrozenSplitError(f"{name}.csv no longer matches its manifest hash")
        return manifest, restored
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        (out_dir / f"{name}.csv").write_bytes(payloads[name])
    manifest_path.write_text(text, encoding="utf-8")  # last: its presence marks a complete write
    return manifest, True


def make_splits(cfg: Config, sample: int | None = None) -> tuple[dict[str, Any], bool]:
    """Raw -> validate -> clean -> (sample) -> split -> leakage checks -> frozen files."""
    frame, _ = clean(load_validated(cfg))
    cleaned_sha256 = hashlib.sha256(_csv_bytes(frame)).hexdigest()
    cleaned_rows = len(frame)
    if sample is not None:
        frame = sample_stratified(frame, sample, cfg.seed)
    splits = stratified_split(frame, cfg.split, cfg.seed)
    check_exact_leakage(splits)
    near = near_duplicate_counts(splits, cfg.split.near_duplicate_threshold)
    out_dir = cfg.split.processed_dir
    if sample is not None:
        out_dir = out_dir / "sample" / f"n{sample}"
    header: dict[str, Any] = {
        "mode": "full" if sample is None else "sample",
        "sample_size": sample,
        "seed": cfg.seed,
        "ratios": {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test},
        "stratify": "intent",
        "source": {
            "hf_revision": cfg.data.hf_revision,
            "raw_sha256": file_sha256(cfg.data.raw_path),
            "cleaned_rows": cleaned_rows,
            "cleaned_sha256": cleaned_sha256,
        },
        "leakage": {
            "exact_shared_normalized_texts": 0,
            "near_duplicates": {
                "rule": NEAR_DUPLICATE_RULE,
                "threshold": cfg.split.near_duplicate_threshold,
                **near,
            },
        },
    }
    return write_frozen_splits(splits, out_dir, header)
