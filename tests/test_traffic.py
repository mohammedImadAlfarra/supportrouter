"""Phase 11: seeded, disjoint traffic batches drawn from validation only."""

import dataclasses
import re
from collections.abc import Sequence

import pandas as pd
import pytest

from supportrouter.config import Config, load_config
from supportrouter.monitoring.traffic import BATCH_NAMES, build_batches, send_batches

BASE = load_config()
CFG = dataclasses.replace(BASE, monitoring=dataclasses.replace(BASE.monitoring, batch_size=60))
ID = re.compile(r"id(\d+)")  # "id" has < 4 letters and digits are never perturbed


def toy_val(per_intent: int = 20) -> pd.DataFrame:
    rows = []
    for intent, category in CFG.labels.intent_to_category.items():
        for _ in range(per_intent):
            i = len(rows)
            text = f"id{i} please help me with something about {intent.replace('_', ' ')}"
            rows.append({"instruction": text, "intent": intent, "category": category})
    return pd.DataFrame(rows)


def ids(texts: Sequence[str]) -> set[int]:
    return {int(m.group(1)) for t in texts if (m := ID.search(t))}


def test_batches_are_disjoint_and_sized() -> None:
    val = toy_val()
    batches = build_batches(val, CFG)
    assert tuple(batches) == BATCH_NAMES
    assert len(batches["reference"]) == len(val) // 2
    assert len(batches["normal"]) == len(batches["shifted"]) == 60
    ref, normal, shifted = (ids(batches[n]) for n in BATCH_NAMES)
    assert len(ref) + len(normal) + len(shifted) == len(val) // 2 + 120  # every id found once
    assert not (ref & normal) and not (ref & shifted) and not (normal & shifted)


def test_shifted_batch_has_the_configured_category_mix_and_typos() -> None:
    val = toy_val()
    batches = build_batches(val, CFG)
    category = val["category"]
    focus = [category[i] in CFG.monitoring.shift_categories for i in ids(batches["shifted"])]
    assert sum(focus) == round(0.70 * 60)
    original = set(val["instruction"])
    assert all(t in original for t in batches["normal"])  # normal traffic is untouched
    changed = sum(t not in original for t in batches["shifted"])
    assert changed > 30  # 6 long words per message at rate 0.20 -> most messages get a typo


def test_traffic_batches_are_reproducible() -> None:
    assert build_batches(toy_val(), CFG) == build_batches(toy_val(), CFG)


def test_not_enough_rows_for_the_shift_is_refused() -> None:
    cfg: Config = dataclasses.replace(
        CFG, monitoring=dataclasses.replace(CFG.monitoring, batch_size=200)
    )
    with pytest.raises(ValueError, match="not enough rows"):
        build_batches(toy_val(), cfg)


def test_send_batches_chunks_to_the_api_limit() -> None:
    sent: list[int] = []

    def post(texts: Sequence[str]) -> str:
        sent.append(len(texts))
        return f"r{len(sent)}"

    out = send_batches({"a": ["x"] * 70, "b": ["y"] * 5}, post, chunk_size=32)
    assert sent == [32, 32, 6, 5]
    assert out == {"a": ["r1", "r2", "r3"], "b": ["r4"]}
    with pytest.raises(ValueError):
        send_batches({"a": ["x"]}, post, chunk_size=0)
