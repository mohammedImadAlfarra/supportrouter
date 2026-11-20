"""Cleaning and dedupe: pure steps, each returning a log entry.

The modelling text is never rewritten here: `normalize` only builds a comparison key.
Typos and extra spaces stay in `instruction` because they are the robustness signal.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleanStep:
    step: str
    reason: str
    rows_before: int
    rows_after: int


class LabelConflictError(ValueError):
    """The same normalized instruction carries more than one intent."""


def normalize(text: str) -> str:
    """Comparison key: lowercase, whitespace collapsed, ends stripped. Shared with Phase 3."""
    return " ".join(text.lower().split())


def check_label_conflicts(frame: pd.DataFrame) -> CleanStep:
    """Fail loudly if one normalized instruction has several intents; never guess a winner."""
    key = frame["instruction"].map(normalize)
    n_intents = frame.groupby(key)["intent"].nunique()
    conflicts = n_intents[n_intents > 1]
    if not conflicts.empty:
        examples = ", ".join(repr(k) for k in conflicts.index[:3])
        raise LabelConflictError(
            f"{len(conflicts)} normalized instructions have more than one intent, e.g. {examples}"
        )
    rows = len(frame)
    return CleanStep("check_label_conflicts", "same text with different intents", rows, rows)


def drop_normalized_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleanStep]:
    """Keep the first row of each normalized instruction, so no text can sit in two splits."""
    key = frame["instruction"].map(normalize)
    out = frame.loc[~key.duplicated(keep="first")].reset_index(drop=True)
    reason = "same normalized instruction"
    return out, CleanStep("drop_normalized_duplicates", reason, len(frame), len(out))


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[CleanStep]]:
    """Run all steps in order; the conflict check runs first so dedupe cannot hide a conflict."""
    log = [check_label_conflicts(frame)]
    frame, step = drop_normalized_duplicates(frame)
    log.append(step)
    return frame, log
