"""Candidate model registry.

Three families, chosen for coverage rather than novelty:

* ``logreg`` -- LogisticRegression: the interpretable linear baseline (kept as
  the permanent fallback family regardless of what wins).
* ``hist_gb`` -- HistGradientBoostingClassifier: sklearn's boosted-trees model.
* ``random_forest`` -- RandomForestClassifier: bagged-trees baseline.

No LightGBM/XGBoost: HistGradientBoosting covers the gradient-boosting case
without an extra dependency. Every estimator is a full ``Pipeline`` that owns
its preprocessing, and every one is seeded for reproducibility.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.config import RANDOM_STATE
from app.ml.preprocessing.pipeline import build_preprocessor


def build_candidate_estimators() -> dict[str, dict]:
    """name -> {estimator, algorithm, hyperparameters}."""
    candidates: dict[str, dict] = {}

    candidates["logreg"] = {
        "estimator": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "algorithm": "LogisticRegression",
        "hyperparameters": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
        },
    }

    candidates["hist_gb"] = {
        "estimator": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=3,
                        max_iter=400,
                        l2_regularization=1.0,
                        min_samples_leaf=25,
                        early_stopping=True,
                        validation_fraction=0.15,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "algorithm": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_depth": 3,
            "max_iter": 400,
            "l2_regularization": 1.0,
            "min_samples_leaf": 25,
            "early_stopping": True,
            "class_weight": "balanced",
        },
    }

    candidates["random_forest"] = {
        "estimator": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 400,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "class_weight": "balanced_subsample",
        },
    }

    return candidates
