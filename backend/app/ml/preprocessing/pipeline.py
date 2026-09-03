"""Single reproducible preprocessing pipeline, shared by training and inference.

Because it is packaged *inside* the fitted estimator (a sklearn ``Pipeline``),
inference applies exactly the transforms that were fitted on the training set --
there is no second copy of this logic anywhere.

* numeric      -> median imputation (+ missing indicator) -> standard scaling
* categorical  -> constant imputation -> one-hot, ``handle_unknown="ignore"``
                  so categories unseen at fit time become an all-zero vector
                  rather than an error.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
