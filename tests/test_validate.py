import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from supportrouter.config import load_config
from supportrouter.data.validate import validate_raw

LABELS = load_config().labels


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flags": ["B", "BLQZ"],
            "instruction": ["cancel my order", "where is my refund"],
            "category": ["ORDER", "REFUND"],
            "intent": ["cancel_order", "track_refund"],
            "response": ["Sure ...", "Let me check ..."],
        },
        dtype="str",
    )


def test_valid_frame_passes_unchanged() -> None:
    frame = valid_frame()
    pd.testing.assert_frame_equal(validate_raw(frame, LABELS), frame)


def test_missing_column_raises() -> None:
    with pytest.raises(SchemaErrors, match="instruction"):
        validate_raw(valid_frame().drop(columns="instruction"), LABELS)


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_raises(text: str) -> None:
    frame = valid_frame()
    frame.loc[0, "instruction"] = text
    with pytest.raises(SchemaErrors, match="empty instruction"):
        validate_raw(frame, LABELS)


def test_unknown_intent_raises() -> None:
    frame = valid_frame()
    frame.loc[0, "intent"] = "buy_spaceship"
    with pytest.raises(SchemaErrors, match="buy_spaceship"):
        validate_raw(frame, LABELS)


def test_unknown_category_raises() -> None:
    frame = valid_frame()
    frame.loc[0, "category"] = "SPACE"
    with pytest.raises(SchemaErrors, match="SPACE"):
        validate_raw(frame, LABELS)


def test_intent_in_wrong_category_raises() -> None:
    frame = valid_frame()
    frame.loc[0, "category"] = "REFUND"
    with pytest.raises(SchemaErrors, match="intent does not belong to its category"):
        validate_raw(frame, LABELS)


def test_extra_column_raises() -> None:
    frame = valid_frame().assign(extra="x")
    with pytest.raises(SchemaErrors, match="extra"):
        validate_raw(frame, LABELS)


def test_config_labels_have_27_intents_and_11_categories() -> None:
    assert len(LABELS.intents) == 27
    assert len(LABELS.categories) == 11
