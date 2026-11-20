import pandas as pd
import pytest

from supportrouter.data.clean import (
    CleanStep,
    LabelConflictError,
    clean,
    drop_normalized_duplicates,
    normalize,
)


def frame(instructions: list[str], intents: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"instruction": instructions, "intent": intents}, dtype="str")


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize("  Do you ship  to\t{{Delivery Country}}? ") == (
        "do you ship to {{delivery country}}?"
    )


def test_dedupe_keeps_first_row_with_its_raw_text() -> None:
    df = frame(
        ["Order {{Order Number}} status", "order  {{order number}} status", "track my order"],
        ["track_order", "track_order", "track_order"],
    )
    out, step = drop_normalized_duplicates(df)
    assert out["instruction"].tolist() == ["Order {{Order Number}} status", "track my order"]
    assert step == CleanStep("drop_normalized_duplicates", "same normalized instruction", 3, 2)


def test_steps_do_not_modify_their_input() -> None:
    df = frame(["a", "A"], ["x", "x"])
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


def test_label_conflict_raises_before_dedupe_can_hide_it() -> None:
    df = frame(["cancel my order", "Cancel my order"], ["cancel_order", "track_order"])
    with pytest.raises(LabelConflictError, match="1 normalized instructions"):
        clean(df)


def test_clean_log_has_counts_for_every_step() -> None:
    df = frame(["a", "A", "b"], ["x", "x", "y"])
    out, log = clean(df)
    assert len(out) == 2
    assert [(s.step, s.rows_before, s.rows_after) for s in log] == [
        ("check_label_conflicts", 3, 3),
        ("drop_normalized_duplicates", 3, 2),
    ]
