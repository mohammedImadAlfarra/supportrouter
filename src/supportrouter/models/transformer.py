"""Transformer T1: DistilBERT fine-tuned for the 27 intents.

Label ids come from the sorted label list in config, never from the data order, so every
run, the Colab run and the API agree on what id 7 means. The tokenizer is pretrained and
only applied (nothing is fitted on data), so there is no train/val leakage here.
The same code runs the CPU SAMPLE run and the Colab FULL run.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from supportrouter.config import TransformerConfig


class IntentDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Tokenized texts plus integer label ids; padding happens per batch in the collator."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.label_ids = [label2id[label] for label in labels]

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "labels": self.label_ids[i],
        }


def label_maps(intents: Sequence[str]) -> tuple[dict[int, str], dict[str, int]]:
    """id -> intent and intent -> id from the config label list (sorted, fixed)."""
    id2label = dict(enumerate(intents))
    return id2label, {label: i for i, label in id2label.items()}


def model_id2label(model: PreTrainedModel) -> dict[int, str]:
    """The id -> intent map stored inside the model config (saved with every checkpoint)."""
    id2label = model.config.id2label
    if not id2label:
        raise ValueError("model config has no id2label map")
    return {int(i): str(label) for i, label in id2label.items()}


def load_pretrained(
    tcfg: TransformerConfig, intents: Sequence[str]
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Pinned tokenizer + DistilBERT with a new, randomly initialised classification head."""
    id2label, label2id = label_maps(intents)
    tokenizer = AutoTokenizer.from_pretrained(tcfg.model_name, revision=tcfg.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg.model_name,
        revision=tcfg.revision,
        num_labels=len(intents),
        id2label=id2label,
        label2id=label2id,
    )
    return tokenizer, model


def _macro_f1(num_labels: int) -> Any:
    """Trainer metric: macro-F1 over every label id, so a never-predicted class counts as 0."""

    def compute(pred: EvalPrediction) -> dict[str, float]:
        y_pred = np.asarray(pred.predictions).argmax(axis=1)
        score = f1_score(
            pred.label_ids, y_pred, labels=list(range(num_labels)), average="macro", zero_division=0
        )
        return {"macro_f1": float(score)}

    return compute


def fit_t1(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_xy: tuple[Sequence[str], Sequence[str]],
    val_xy: tuple[Sequence[str], Sequence[str]],
    tcfg: TransformerConfig,
    seed: int,
    output_dir: Path,
) -> tuple[PreTrainedModel, list[dict[str, float]]]:
    """Fine-tune on train, evaluate on val every epoch, keep the epoch with the best val macro-F1.

    Returns the best model and the per-epoch log history (train loss, val loss, val macro-F1).
    `test.csv` is never passed here.
    """
    set_seed(seed)  # Python, NumPy and torch RNGs
    label2id = {label: i for i, label in model_id2label(model).items()}
    train_ds = IntentDataset(*train_xy, tokenizer, label2id, tcfg.max_length)
    val_ds = IntentDataset(*val_xy, tokenizer, label2id, tcfg.max_length)
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=not torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        learning_rate=tcfg.learning_rate,
        num_train_epochs=tcfg.epochs,
        per_device_train_batch_size=tcfg.batch_size,
        per_device_eval_batch_size=tcfg.eval_batch_size,
        warmup_steps=tcfg.warmup_ratio,  # a float in [0, 1) is a ratio of all steps (v5)
        weight_decay=tcfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,  # best checkpoint is always kept on top of this limit
        save_only_model=True,
        report_to="none",  # train.py logs one MLflow run itself; no auto-created runs
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_macro_f1(len(label2id)),
    )
    trainer.train()
    best = trainer.model  # the best val epoch, reloaded by `load_best_model_at_end`
    if not isinstance(best, PreTrainedModel):
        raise TypeError(f"unexpected model type after training: {type(best).__name__}")
    history = [
        {k: float(v) for k, v in entry.items() if isinstance(v, int | float)}
        for entry in trainer.state.log_history
    ]
    return best, history


def save_t1(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, model_dir: Path) -> Path:
    """Save weights, config (with id2label) and tokenizer, plus one SHA-256 per file.

    The sidecar `<model_dir>.sha256` uses the `sha256sum` format: `<hex>  <relative path>`.
    """
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    lines = [f"{digest}  {name}\n" for name, digest in dir_sha256(model_dir).items()]
    sidecar = model_dir.with_name(model_dir.name + ".sha256")
    sidecar.write_text("".join(lines), encoding="utf-8")
    return sidecar


def dir_sha256(model_dir: Path) -> dict[str, str]:
    """Relative path -> SHA-256 hex of every file under `model_dir`, sorted by path."""
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    return {
        p.relative_to(model_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def checksum_mismatches(model_dir: Path, sidecar_text: str) -> list[str]:
    """Files that are missing, extra or changed compared with a `sha256sum`-style sidecar."""
    expected: dict[str, str] = {}
    for line in sidecar_text.splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            expected[name] = digest
    actual = dir_sha256(model_dir) if model_dir.is_dir() else {}
    return sorted(n for n in expected.keys() | actual.keys() if expected.get(n) != actual.get(n))


def load_t1(model_dir: Path) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a saved T1 folder. Call only after `checksum_mismatches` returned no files.

    Weights are `safetensors` (plain tensors, no pickle), so loading runs no stored code.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


@torch.no_grad()
def predict_with_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    batch_size: int = 64,
) -> tuple[list[str], list[float]]:
    """Shared interface: top intent and its softmax probability."""
    model.train(False)  # eval mode: dropout off
    id2label = model_id2label(model)
    labels: list[str] = []
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[start : start + batch_size]),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        proba = torch.softmax(model(**batch).logits, dim=-1)
        best = proba.max(dim=-1)
        labels += [id2label[int(i)] for i in best.indices]
        scores += [float(p) for p in best.values]
    return labels, scores
