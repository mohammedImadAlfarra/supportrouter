"""Simulated API traffic for the drift check (Phase 11).

Three disjoint batches are drawn from the validation split only (the test split stays frozen),
all seeded:
- `reference`: one stratified half of val, the "known good" traffic the model was built for;
- `normal`:   `batch_size` random rows of the other half (same distribution, no drift expected);
- `shifted`:  `batch_size` other rows of that half, `shift_share` of them from
  `shift_categories` (a prior shift in what customers ask about), then synthetic typos at
  `shift_typo_rate` (a covariate shift in how they write it).
Only `instruction` is sent; `response` is never an input.
"""

import random
from collections.abc import Callable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from supportrouter.config import Config
from supportrouter.evaluate import perturb_text

BATCH_NAMES = ("reference", "normal", "shifted")


def build_batches(val: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Return the three batches of message texts, in send order."""
    mon, seed = cfg.monitoring, cfg.seed
    reference, traffic = train_test_split(
        val, train_size=mon.reference_share, stratify=val["intent"], random_state=seed
    )
    normal = traffic.sample(n=mon.batch_size, random_state=seed)
    rest = traffic.drop(index=normal.index)

    in_focus = rest["category"].isin(mon.shift_categories)
    n_focus = round(mon.shift_share * mon.batch_size)
    n_other = mon.batch_size - n_focus
    if in_focus.sum() < n_focus or (~in_focus).sum() < n_other:
        raise ValueError(
            f"not enough rows for the shifted batch: need {n_focus} in {mon.shift_categories} "
            f"(have {int(in_focus.sum())}) and {n_other} outside (have {int((~in_focus).sum())})"
        )
    shifted = pd.concat(
        [
            rest[in_focus].sample(n=n_focus, random_state=seed),
            rest[~in_focus].sample(n=n_other, random_state=seed),
        ]
    ).sample(frac=1.0, random_state=seed)  # mix the two parts so the send order is not sorted

    rng = random.Random(seed)
    min_len = cfg.evaluation.typo_min_word_len
    shifted_texts = [
        perturb_text(t, mon.shift_typo_rate, rng, min_len) for t in shifted["instruction"]
    ]
    return {
        "reference": reference["instruction"].tolist(),
        "normal": normal["instruction"].tolist(),
        "shifted": shifted_texts,
    }


def send_batches(
    batches: dict[str, list[str]],
    post: Callable[[Sequence[str]], str],
    chunk_size: int,
) -> dict[str, list[str]]:
    """Send every batch in chunks of at most `chunk_size` texts (the API's `max_batch`).

    `post` sends one request and returns its `request_id`; it is injected so tests run without
    a server. The returned ids are the only link between a batch and its log lines (no text).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    ids: dict[str, list[str]] = {}
    for name, texts in batches.items():
        ids[name] = [post(texts[i : i + chunk_size]) for i in range(0, len(texts), chunk_size)]
    return ids
