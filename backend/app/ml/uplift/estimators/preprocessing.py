"""Preprocessors for the uplift meta-learners.

Both reuse the predictive layer's transform choices (median-impute + indicator +
scale for numerics; constant-impute + one-hot ``handle_unknown='ignore'`` for
categoricals) so a row scored by the uplift models is transformed identically to
one scored by the predictive model.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.preprocessing.pipeline import build_preprocessor as build_s_learner_preprocessor
from app.ml.uplift.features import (
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
)

__all__ = ["build_s_learner_preprocessor", "build_baseline_preprocessor"]


def build_baseline_preprocessor() -> ColumnTransformer:
    """Preprocessor over ``BASELINE_FEATURES`` (``ALL_FEATURES`` minus
    ``action``) -- used by the control response model and each per-action
    T-learner, where ``action`` is constant within the training slice."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, BASELINE_NUMERIC_FEATURES),
            ("cat", categorical, BASELINE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
