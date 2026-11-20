import hashlib
from pathlib import Path

import pandas as pd
import pytest

from supportrouter.config import SplitConfig
from supportrouter.data.load import MANIFEST_NAME, file_sha256
from supportrouter.data.split import (
    FrozenSplitError,
    LeakageError,
    check_exact_leakage,
    near_duplicate_counts,
    sample_stratified,
    stratified_split,
    write_frozen_splits,
)

CFG = SplitConfig(Path("unused"), train=0.70, val=0.15, test=0.15, near_duplicate_threshold=0.9)
INTENTS = {"cancel_order": 40, "track_order": 60, "get_refund": 100}


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = [(f"message {i} about {x}", x) for x, n in INTENTS.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["instruction", "intent"], dtype="str")


def split_hash(splits: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for name, part in splits.items():
        digest.update(name.encode())
        digest.update(part.to_csv(index=False).encode())
    return digest.hexdigest()


def test_same_seed_gives_same_split_hash(frame: pd.DataFrame) -> None:
    assert split_hash(stratified_split(frame, CFG, 42)) == split_hash(
        stratified_split(frame, CFG, 42)
    )
    assert split_hash(stratified_split(frame, CFG, 42)) != split_hash(
        stratified_split(frame, CFG, 7)
    )


def test_split_output_has_no_shared_text(frame: pd.DataFrame) -> None:
    check_exact_leakage(stratified_split(frame, CFG, 42))


def test_leakage_check_catches_normalized_duplicate(frame: pd.DataFrame) -> None:
    splits = stratified_split(frame, CFG, 42)
    row = splits["train"].iloc[[0]]
    leaked = row.assign(instruction="  " + row["instruction"].str.upper())
    splits["test"] = pd.concat([splits["test"], leaked], ignore_index=True)
    with pytest.raises(LeakageError, match="1 normalized instructions .* both train and test"):
        check_exact_leakage(splits)


def test_split_sizes_and_stratification(frame: pd.DataFrame) -> None:
    splits = stratified_split(frame, CFG, 42)
    sizes = {name: len(part) for name, part in splits.items()}
    assert sizes == {"train": 140, "val": 30, "test": 30}
    assert sum(len(p) for p in splits.values()) == len(frame)
    overall = frame["intent"].value_counts(normalize=True)
    for part in splits.values():
        share = part["intent"].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.02


def test_split_does_not_modify_input(frame: pd.DataFrame) -> None:
    before = frame.copy()
    stratified_split(frame, CFG, 42)
    pd.testing.assert_frame_equal(frame, before)


def text_frame(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"instruction": texts, "intent": "x"}, dtype="str")


def test_near_duplicates_count_typo_variants_not_other_texts() -> None:
    # Measured cosines to the nearest train row: 0.927, 0.848, 0.187 | 1.0, 0.0
    splits = {
        "train": text_frame(["i want to cancel my order please", "where is my refund"]),
        "val": text_frame(
            [
                "I want to cancel my ordr please",
                "i want to cancel my subscription please",
                "how do i change my address",
            ]
        ),
        "test": text_frame(["Where is my refund", "set up a new account"]),
    }
    assert near_duplicate_counts(splits, threshold=0.9) == {"val": 1, "test": 1}
    assert near_duplicate_counts(splits, threshold=0.8) == {"val": 2, "test": 1}


def test_sample_is_stratified_and_seeded(frame: pd.DataFrame) -> None:
    sample = sample_stratified(frame, 100, 42)
    assert sample["intent"].value_counts().to_dict() == {
        "get_refund": 50,
        "track_order": 30,
        "cancel_order": 20,
    }
    pd.testing.assert_frame_equal(sample, sample_stratified(frame, 100, 42))


def test_rerun_verifies_frozen_files_without_rewriting(frame: pd.DataFrame, tmp_path: Path) -> None:
    splits = stratified_split(frame, CFG, 42)
    manifest, written = write_frozen_splits(splits, tmp_path, {"seed": 42})
    assert written
    for name in ("train", "val", "test"):
        assert file_sha256(tmp_path / f"{name}.csv") == manifest["splits"][name]["sha256"]
    mtimes = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}

    again, written = write_frozen_splits(stratified_split(frame, CFG, 42), tmp_path, {"seed": 42})
    assert not written
    assert again == manifest
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()} == mtimes


def test_frozen_split_refuses_a_different_split(frame: pd.DataFrame, tmp_path: Path) -> None:
    write_frozen_splits(stratified_split(frame, CFG, 42), tmp_path, {"seed": 42})
    before = (tmp_path / MANIFEST_NAME).read_bytes()
    with pytest.raises(FrozenSplitError, match="frozen split is refused"):
        write_frozen_splits(stratified_split(frame, CFG, 7), tmp_path, {"seed": 7})
    assert (tmp_path / MANIFEST_NAME).read_bytes() == before


def test_frozen_split_detects_an_edited_file(frame: pd.DataFrame, tmp_path: Path) -> None:
    splits = stratified_split(frame, CFG, 42)
    write_frozen_splits(splits, tmp_path, {"seed": 42})
    with (tmp_path / "test.csv").open("a", encoding="utf-8") as f:
        f.write("extra row,cancel_order\n")
    with pytest.raises(FrozenSplitError, match="test.csv no longer matches"):
        write_frozen_splits(splits, tmp_path, {"seed": 42})


def test_missing_files_are_restored_from_the_committed_manifest(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    # CI has only the committed manifest; rebuilding from the same raw data must
    # reproduce the frozen files byte for byte, and the manifest is never rewritten.
    manifest, _ = write_frozen_splits(stratified_split(frame, CFG, 42), tmp_path, {"seed": 42})
    manifest_bytes = (tmp_path / MANIFEST_NAME).read_bytes()
    for name in ("train", "val", "test"):
        (tmp_path / f"{name}.csv").unlink()

    again, written = write_frozen_splits(stratified_split(frame, CFG, 42), tmp_path, {"seed": 42})
    assert written
    assert again == manifest
    assert (tmp_path / MANIFEST_NAME).read_bytes() == manifest_bytes
    for name in ("train", "val", "test"):
        assert file_sha256(tmp_path / f"{name}.csv") == manifest["splits"][name]["sha256"]


def test_missing_files_are_not_restored_from_a_different_split(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    write_frozen_splits(stratified_split(frame, CFG, 42), tmp_path, {"seed": 42})
    (tmp_path / "val.csv").unlink()
    with pytest.raises(FrozenSplitError, match="frozen split is refused"):
        write_frozen_splits(stratified_split(frame, CFG, 7), tmp_path, {"seed": 42})
    assert not (tmp_path / "val.csv").exists()
