import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from supportrouter.config import Config, load_config
from supportrouter.data.load import SplitHashError, load_split
from supportrouter.data.split import write_frozen_splits
from supportrouter.models.baseline import build_b0, build_b1, predict_with_scores

CFG = load_config()
TRAIN = pd.DataFrame(
    {
        "instruction": [
            "cancel my order",
            "please cancel the order",
            "where is my refund",
            "track my refund",
            "refund status please",
        ],
        "intent": ["cancel_order", "cancel_order", "track_refund", "track_refund", "track_refund"],
    }
)


def test_b1_vocabulary_comes_from_train_only() -> None:
    model = build_b1(CFG.baselines, c=1.0, seed=42).fit(TRAIN["instruction"], TRAIN["intent"])
    vocab = model.named_steps["tfidf"].vocabulary_
    assert "refund" in vocab and "cancel the" in vocab  # unigram + bigram from train
    model.predict(["I want my invoice"])  # a val-like text with unseen words
    assert "invoice" not in model.named_steps["tfidf"].vocabulary_


def test_b0_always_predicts_the_majority_intent() -> None:
    model = build_b0().fit(TRAIN["instruction"], TRAIN["intent"])
    labels, scores = predict_with_scores(model, ["cancel my order", "anything"])
    assert labels == ["track_refund", "track_refund"]
    # "most_frequent" gives a one-hot predict_proba (only "prior" returns 0.6/0.4),
    # so B0's score is always 1.0: it is a floor for metrics, never a served model.
    assert scores == [1.0, 1.0]


def test_predict_with_scores_matches_predict() -> None:
    model = build_b1(CFG.baselines, c=10.0, seed=42).fit(TRAIN["instruction"], TRAIN["intent"])
    texts = ["cancel order now", "refund where"]
    labels, scores = predict_with_scores(model, texts)
    assert labels == list(model.predict(texts)) == ["cancel_order", "track_refund"]
    assert all(0.5 < s <= 1.0 for s in scores)


def split_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flags": ["B"] * n,
            "instruction": [f"cancel order {i}" for i in range(n)],
            "category": ["ORDER"] * n,
            "intent": ["cancel_order"] * n,
            "response": ["Sure"] * n,
        },
        dtype="str",
    )


@pytest.fixture
def split_cfg(tmp_path: Path) -> Config:
    splits = {"train": split_frame(4), "val": split_frame(2), "test": split_frame(2)}
    write_frozen_splits(splits, tmp_path, {"seed": 42})
    return dataclasses.replace(CFG, split=dataclasses.replace(CFG.split, processed_dir=tmp_path))


def test_load_split_returns_validated_frame(split_cfg: Config) -> None:
    frame = load_split(split_cfg, "train")
    assert len(frame) == 4
    assert list(frame.columns) == ["flags", "instruction", "category", "intent", "response"]


def test_load_split_refuses_a_file_that_differs_from_the_manifest(split_cfg: Config) -> None:
    path = split_cfg.split.processed_dir / "val.csv"
    path.write_text(path.read_text().replace("cancel order 1", "cancel order 9"))
    with pytest.raises(SplitHashError, match="val.csv: sha256"):
        load_split(split_cfg, "val")
