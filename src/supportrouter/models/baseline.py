"""Baselines B0 and B1.

B0 is the sanity floor: it always predicts the most frequent intent of train.
B1 is one sklearn `Pipeline`, so TF-IDF is fitted inside `fit` on train only.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from supportrouter.config import BaselineConfig


def build_b0() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def build_b1(cfg: BaselineConfig, c: float, seed: int) -> Pipeline:
    """TF-IDF word n-grams + multinomial Logistic Regression with regularization strength `c`."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=cfg.ngram_range, lowercase=True)),
            ("clf", LogisticRegression(C=c, max_iter=cfg.max_iter, random_state=seed)),
        ]
    )


def predict_with_scores(
    model: DummyClassifier | Pipeline, texts: Sequence[str]
) -> tuple[list[str], list[float]]:
    """Shared interface: top intent and its probability per text.

    The probability is what the API compares with the review threshold later.
    """
    proba = np.asarray(model.predict_proba(list(texts)))
    best = proba.argmax(axis=1)
    classes = model.classes_
    return [str(classes[i]) for i in best], [float(p) for p in proba.max(axis=1)]
