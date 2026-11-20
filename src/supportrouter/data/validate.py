"""Raw data schema: fail loudly when the Bitext file changes shape or labels."""

import pandas as pd
import pandera.pandas as pa

from supportrouter.config import LabelConfig


def _not_blank(series: pd.Series) -> pd.Series:
    return series.str.strip().str.len() > 0


def raw_schema(labels: LabelConfig) -> pa.DataFrameSchema:
    """Schema of the raw CSV as read by `read_raw` (every column is text).

    `strict=True` refuses extra columns; `coerce=False` never converts values silently.
    """
    mapping = dict(labels.intent_to_category)
    return pa.DataFrameSchema(
        {
            "flags": pa.Column(str, pa.Check.str_matches(r"^[A-Z]+$")),
            "instruction": pa.Column(str, pa.Check(_not_blank, error="empty instruction")),
            "category": pa.Column(str, pa.Check.isin(labels.categories)),
            "intent": pa.Column(str, pa.Check.isin(labels.intents)),
            "response": pa.Column(str),
        },
        checks=pa.Check(
            lambda df: df["intent"].map(mapping) == df["category"],
            error="intent does not belong to its category",
        ),
        strict=True,
        coerce=False,
    )


def validate_raw(frame: pd.DataFrame, labels: LabelConfig) -> pd.DataFrame:
    """Validate and return the frame unchanged; raise `SchemaErrors` listing every failure."""
    return raw_schema(labels).validate(frame, lazy=True)
