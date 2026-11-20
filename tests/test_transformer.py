"""T1 tests on a tiny, randomly initialised DistilBERT: no download, a few seconds each."""

import dataclasses
from pathlib import Path

import pytest
import torch
from sklearn.metrics import f1_score
from transformers import DistilBertConfig, DistilBertForSequenceClassification, set_seed
from transformers import DistilBertTokenizer as Tokenizer

from supportrouter.config import load_config
from supportrouter.models.transformer import fit_t1, label_maps, predict_with_scores

INTENTS = ("cancel_order", "get_invoice", "track_refund")  # sorted, like the config list
WORDS = {
    "cancel_order": "cancel stop order purchase",
    "get_invoice": "invoice bill receipt copy",
    "track_refund": "refund money back status",
}
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# A 1-layer random model learns the toy data only partly in a few steps (measured: val
# macro-F1 0.17 -> 0.22); the tests check the wiring and best-epoch choice, not quality.
TCFG = dataclasses.replace(
    load_config().transformer, epochs=4, batch_size=8, learning_rate=5e-3, warmup_ratio=0.0
)


def frame(n: int, offset: int) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for intent, words in WORDS.items():
        w = words.split()
        for i in range(offset, offset + n):
            texts.append(f"{w[i % 4]} {w[(i + 1) % 4]} please")
            labels.append(intent)
    return texts, labels


def tiny_tokenizer() -> Tokenizer:
    words = sorted({w for ws in WORDS.values() for w in ws.split()} | {"please"})
    return Tokenizer(vocab={w: i for i, w in enumerate(SPECIAL + words)})


def tiny_model(seed: int = 42) -> DistilBertForSequenceClassification:
    set_seed(seed)
    id2label, label2id = label_maps(INTENTS)
    config = DistilBertConfig(
        vocab_size=len(SPECIAL) + 13,
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=len(INTENTS),
        id2label=id2label,
        label2id=label2id,
    )
    return DistilBertForSequenceClassification(config)


def run_fit(tmp_path: Path) -> tuple[DistilBertForSequenceClassification, list[dict[str, float]]]:
    tok = tiny_tokenizer()
    train_x, train_y = frame(16, 0)
    # Reverse the data order: label ids must still follow the config list, not the data.
    return fit_t1(
        tiny_model(), tok, (train_x[::-1], train_y[::-1]), frame(3, 50), TCFG, 42, tmp_path
    )


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> tuple:  # type: ignore[type-arg]
    return run_fit(tmp_path_factory.mktemp("t1"))


def test_label_ids_follow_config_order_not_data_order(fitted: tuple) -> None:  # type: ignore[type-arg]
    model, _ = fitted
    assert model.config.id2label == {0: "cancel_order", 1: "get_invoice", 2: "track_refund"}


def test_one_val_evaluation_per_epoch_and_best_epoch_is_returned(fitted: tuple) -> None:  # type: ignore[type-arg]
    model, history = fitted
    val_f1 = [h["eval_macro_f1"] for h in history if "eval_macro_f1" in h]
    assert len(val_f1) == TCFG.epochs
    val_x, val_y = frame(3, 50)
    labels, _ = predict_with_scores(model, tiny_tokenizer(), val_x, TCFG.max_length)
    returned_f1 = f1_score(val_y, labels, labels=list(INTENTS), average="macro", zero_division=0)
    assert returned_f1 == pytest.approx(max(val_f1))


def test_same_seed_gives_same_t1_predictions(tmp_path: Path, fitted: tuple) -> None:  # type: ignore[type-arg]
    first, _ = fitted
    second, _ = run_fit(tmp_path)
    texts = frame(3, 50)[0]
    _, s1 = predict_with_scores(first, tiny_tokenizer(), texts, TCFG.max_length)
    _, s2 = predict_with_scores(second, tiny_tokenizer(), texts, TCFG.max_length)
    assert torch.allclose(torch.tensor(s1), torch.tensor(s2), atol=1e-6)


def test_predict_returns_known_labels_and_batching_does_not_change_scores(fitted: tuple) -> None:  # type: ignore[type-arg]
    model, _ = fitted
    texts = frame(3, 50)[0] + ["completely unknown words here"]
    labels, scores = predict_with_scores(model, tiny_tokenizer(), texts, TCFG.max_length, 64)
    _, scores_b2 = predict_with_scores(model, tiny_tokenizer(), texts, TCFG.max_length, 2)
    assert set(labels) <= set(INTENTS) and len(labels) == len(texts)
    assert all(0 < s <= 1 for s in scores)
    assert torch.allclose(torch.tensor(scores), torch.tensor(scores_b2), atol=1e-5)
