"""Common interface for the uplift meta-learners.

Every estimator answers the same three questions for a context frame ``X``:

    mu_0(X)      -> P(recovery | X, no intervention)
    mu_a(X)      -> P(recovery | X, action = a)     for each treatment action a
    uplift_a(X)  -> mu_a(X) - mu_0(X)

Exact definitions and the identification assumptions are in
``app/ml/uplift/README.md``. These are *conditional average* effects estimated
from randomized synthetic data -- never individual causal effects.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.uplift.config import TREATMENT_ACTIONS


@dataclass
class UpliftPrediction:
    baseline_probability: float
    treatment_probability: dict[str, float]
    uplift: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "baseline_probability": self.baseline_probability,
            "treatment_probability": self.treatment_probability,
            "uplift": self.uplift,
        }


class BaseUpliftEstimator(abc.ABC):
    """Fit on an :class:`~app.ml.uplift.dataset.UpliftDataset` frame; predict on
    a shared point-in-time feature frame (any number of rows)."""

    learner_type: str = "base"

    def __init__(self, *, actions: list[str] | None = None) -> None:
        self.actions = list(actions or TREATMENT_ACTIONS)
        self._fitted = False

    # -- fitting --------------------------------------------------
    @abc.abstractmethod
    def fit(self, frame: pd.DataFrame) -> "BaseUpliftEstimator": ...

    # -- scoring -------------------------------------------------
    @abc.abstractmethod
    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray: ...

    @abc.abstractmethod
    def predict_action(self, X: pd.DataFrame, action: str) -> np.ndarray: ...

    def predict_uplift(self, X: pd.DataFrame, action: str) -> np.ndarray:
        return self.predict_action(X, action) - self.predict_baseline(X)

    def predict_all(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        base = self.predict_baseline(X)
        out: dict[str, np.ndarray] = {"none": base}
        for a in self.actions:
            out[a] = self.predict_action(X, a)
        return out

    def predict_row(self, X: pd.DataFrame) -> UpliftPrediction:
        if len(X) != 1:
            raise ValueError("predict_row expects exactly one row")
        base = float(self.predict_baseline(X)[0])
        treat = {a: float(self.predict_action(X, a)[0]) for a in self.actions}
        return UpliftPrediction(
            baseline_probability=base,
            treatment_probability=treat,
            uplift={a: treat[a] - base for a in self.actions},
        )

    # -- introspection ----------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted")


@dataclass
class EstimatorMetadata:
    learner_type: str
    base_algorithm: str
    calibration_method: str
    calibration_cv: int
    per_arm_training_rows: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
