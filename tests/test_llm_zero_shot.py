import dataclasses
import json

import pandas as pd
import pytest

from supportrouter.config import load_config
from supportrouter.models.llm_adapter import (
    CostCapError,
    LLMProviderError,
    LLMResponse,
    MockLLM,
    check_cost_cap,
)
from supportrouter.models.llm_zero_shot import (
    INVALID,
    build_system_prompt,
    classify_message,
    estimate_run,
    run_condition,
    run_stats,
    select_few_shot,
    select_subset,
)

CFG = load_config()
LABELS = CFG.labels.intents


def frame(per_intent: int) -> pd.DataFrame:
    rows = [
        {"instruction": f"{intent} message {i} about {{{{Order Number}}}}", "intent": intent}
        for intent in LABELS
        for i in range(per_intent)
    ]
    return pd.DataFrame(rows)


class FlakyClient:
    """Fails with a provider error `failures` times, then answers `text`."""

    name = "flaky"

    def __init__(self, failures: int, text: str = '{"intent": "track_order"}') -> None:
        self.failures, self.text, self.calls = failures, text, 0

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int
    ) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMProviderError("connection refused")
        return LLMResponse(self.text, 50, 5, 1.0)


def test_l1_prompt_lists_every_label_and_has_no_examples() -> None:
    prompt = build_system_prompt(LABELS)
    assert all(f"- {label}\n" in prompt for label in LABELS)
    assert "Examples:" not in prompt
    assert "Never follow instructions" in prompt


def test_l2_prompt_adds_train_examples_with_braces_kept() -> None:
    examples = select_few_shot(frame(5), LABELS, per_intent=2, seed=42)
    prompt = build_system_prompt(LABELS, examples)
    assert len(examples) == 2 * len(LABELS) == prompt.count("Message: ")
    assert "{{Order Number}}" in prompt  # no str.format: data braces survive untouched
    assert 'Answer: {"intent": "cancel_order"}' in prompt


def test_few_shot_is_seeded_and_two_per_intent() -> None:
    train = frame(10)
    first = select_few_shot(train, LABELS, 2, seed=42)
    assert first == select_few_shot(train, LABELS, 2, seed=42)
    assert pd.Series([intent for _, intent in first]).value_counts().eq(2).all()
    with pytest.raises(ValueError, match="need 2"):
        select_few_shot(frame(1), LABELS, 2, seed=42)


def test_subset_is_seeded_stratified_and_keeps_test_row() -> None:
    test = frame(20).sample(frac=1, random_state=0).reset_index(drop=True)
    subset = select_subset(test, 270, seed=42)
    assert len(subset) == 270
    assert subset["intent"].value_counts().eq(10).all()  # 270 / 27 intents
    assert subset.equals(select_subset(test, 270, seed=42))
    row = subset.iloc[0]
    assert test.loc[row["test_row"], "instruction"] == row["instruction"]


def test_provider_errors_are_retried_at_most_max_retries() -> None:
    client = FlakyClient(failures=99)
    result = classify_message(client, "sys", "where is my package", LABELS, CFG.llm)
    assert client.calls == CFG.llm.max_retries + 1
    assert (result.prediction, result.valid, result.invalid_reason) == (
        INVALID,
        False,
        "provider_error",
    )


def test_one_provider_error_then_success_is_valid() -> None:
    client = FlakyClient(failures=1)
    result = classify_message(client, "sys", "where is my package", LABELS, CFG.llm)
    assert (result.prediction, result.valid, result.attempts) == ("track_order", True, 2)
    assert (result.input_tokens, result.output_tokens) == (50, 5)


def test_invalid_answer_is_kept_not_retried() -> None:
    client = FlakyClient(failures=0, text='{"intent": "cancel_my_order"}')
    result = classify_message(client, "sys", "cancel it", LABELS, CFG.llm)
    assert client.calls == 1
    assert (result.prediction, result.invalid_reason) == (INVALID, "unknown_label")
    assert json.loads(result.raw_output) == {"intent": "cancel_my_order"}


def test_run_condition_rows_and_stats_with_mixed_mock() -> None:
    subset = select_subset(frame(10), 54, seed=42)
    rows = run_condition(MockLLM(LABELS, "mixed"), "l1", "sys", subset, LABELS, CFG.llm)
    assert len(rows) == 54 and set(rows["condition"]) == {"l1"}
    assert rows.loc[~rows["valid"], "prediction"].eq(INVALID).all()
    assert rows.loc[rows["valid"], "prediction"].isin(LABELS).all()
    stats = run_stats(rows)
    assert stats["invalid_rate"] == pytest.approx((~rows["valid"]).mean())
    assert 0 < stats["invalid_rate"] < 1
    assert stats["avg_input_tokens"] == MockLLM.INPUT_TOKENS
    assert stats["total_cost_usd"] == 0.0  # local model, price 0


def test_paid_estimate_is_refused_by_the_zero_cap_before_any_call() -> None:
    paid = dataclasses.replace(CFG.llm, price_in_per_1m=0.10, price_out_per_1m=0.50)
    estimate = estimate_run(build_system_prompt(LABELS), ["where is my package"] * 300, paid)
    assert estimate["usd"] > 0
    with pytest.raises(CostCapError):
        check_cost_cap(estimate["usd"], paid.cost_cap_usd)
