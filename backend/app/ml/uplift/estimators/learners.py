"""S-learner and T-learner implementations (sklearn-compatible, no new deps).

**T-learner** -- one calibrated classifier per arm:

    mu_0  <- fit on control rows            (features: BASELINE_FEATURES)
    mu_a  <- fit on rows with action == a   (features: BASELINE_FEATURES)
    uplift_a(X) = mu_a(X) - mu_0(X)

Pro: each model is free to use a completely different response surface per arm.
Con: the tiny control arm gets its own small model -> high variance in mu_0.

**S-learner** -- a single calibrated classifier over *all* rows with ``action``
(including the ``'none'`` control value) as an input feature
(``S_LEARNER_FEATURES == ALL_FEATURES``):

    mu_a(X) = model.predict_proba([X | action = a])
    mu_0(X) = model.predict_proba([X | action = 'none'])

Pro: pools all rows, so mu_0 borrows strength from the treatment arms; lower
variance. Con: can *regularize away* a small treatment effect (the classic
S-learner bias toward zero uplift).

Comparing the two on held-out Qini / policy value is Phase 11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.features.schema import ALL_FEATURES, LABEL
from app.ml.uplift.config import CONTROL_ACTION, RANDOM_STATE
from app.ml.uplift.estimators.base import BaseUpliftEstimator, EstimatorMetadata
from app.ml.uplift.estimators.preprocessing import (
    build_baseline_preprocessor,
    build_s_learner_preprocessor,
)
from app.ml.uplift.features import BASELINE_FEATURES, TREATMENT_FEATURE

_CALIBRATION_CV = 3  # the control arm is small; 3 keeps >= ~15 positives / fold


def _base_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=RANDOM_STATE,
        ), "LogisticRegression"
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=300,
            l2_regularization=1.0, min_samples_leaf=25,
            early_stopping=True, validation_fraction=0.15,
            class_weight="balanced", random_state=RANDOM_STATE,
        ), "HistGradientBoostingClassifier"
    raise ValueError(f"unknown base classifier {name!r}")


def _calibrated(preprocessor, clf, X, y, *, cv: int) -> CalibratedClassifierCV:
    pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    # A stratified fold must hold >= 2 of each class; shrink cv if the slice is
    # tiny rather than crashing.
    n_pos = int(np.asarray(y).sum())
    n_neg = int(len(y) - n_pos)
    safe_cv = max(2, min(cv, n_pos, n_neg))
    model = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=safe_cv)
    model.fit(X, y)
    return model, safe_cv


class TLearner(BaseUpliftEstimator):
    learner_type = "t_learner"

    def __init__(self, *, base: str = "hist_gb", actions: list[str] | None = None) -> None:
        super().__init__(actions=actions)
        self.base = base
        self._models: dict[str, CalibratedClassifierCV] = {}
        self.metadata: EstimatorMetadata | None = None

    def fit(self, frame: pd.DataFrame) -> "TLearner":
        rows: dict[str, int] = {}
        cvs: list[int] = []
        for arm in [CONTROL_ACTION, *self.actions]:
            sub = frame[frame["arm"] == arm]
            rows[arm] = int(len(sub))
            if sub.empty or sub[LABEL].nunique() < 2:
                raise ValueError(f"arm {arm!r}: need both classes, got {len(sub)} rows")
            clf, algo = _base_classifier(self.base)
            model, used_cv = _calibrated(
                build_baseline_preprocessor(), clf,
                sub[BASELINE_FEATURES], sub[LABEL].astype(int), cv=_CALIBRATION_CV,
            )
            self._models[arm] = model
            cvs.append(used_cv)
        self._fitted = True
        self.metadata = EstimatorMetadata(
            learner_type=self.learner_type,
            base_algorithm=algo,
            calibration_method="sigmoid",
            calibration_cv=min(cvs),
            per_arm_training_rows=rows,
            notes=[
                "one calibrated classifier per arm over BASELINE_FEATURES",
                f"control arm trained on {rows[CONTROL_ACTION]} rows",
            ],
        )
        return self

    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._models[CONTROL_ACTION].predict_proba(X[BASELINE_FEATURES])[:, 1]

    def predict_action(self, X: pd.DataFrame, action: str) -> np.ndarray:
        self._check_fitted()
        if action not in self._models:
            raise ValueError(f"unknown action {action!r}")
        return self._models[action].predict_proba(X[BASELINE_FEATURES])[:, 1]


class SLearner(BaseUpliftEstimator):
    learner_type = "s_learner"

    def __init__(self, *, base: str = "hist_gb", actions: list[str] | None = None) -> None:
        super().__init__(actions=actions)
        self.base = base
        self._model: CalibratedClassifierCV | None = None
        self.metadata: EstimatorMetadata | None = None

    def fit(self, frame: pd.DataFrame) -> "SLearner":
        clf, algo = _base_classifier(self.base)
        model, used_cv = _calibrated(
            build_s_learner_preprocessor(), clf,
            frame[ALL_FEATURES], frame[LABEL].astype(int), cv=_CALIBRATION_CV,
        )
        self._model = model
        self._fitted = True
        rows = frame["arm"].value_counts().to_dict()
        self.metadata = EstimatorMetadata(
            learner_type=self.learner_type,
            base_algorithm=algo,
            calibration_method="sigmoid",
            calibration_cv=used_cv,
            per_arm_training_rows={k: int(v) for k, v in rows.items()},
            notes=[
                "single calibrated classifier over ALL_FEATURES with 'action' "
                "as the treatment indicator ('none' == control)",
                "known S-learner risk: regularization can shrink small uplifts "
                "toward zero -- cross-check against the T-learner",
            ],
        )
        return self

    def _score_with_action(self, X: pd.DataFrame, action: str) -> np.ndarray:
        self._check_fitted()
        Xa = X[ALL_FEATURES].copy()
        Xa[TREATMENT_FEATURE] = action
        return self._model.predict_proba(Xa)[:, 1]

    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        return self._score_with_action(X, CONTROL_ACTION)

    def predict_action(self, X: pd.DataFrame, action: str) -> np.ndarray:
        if action not in self.actions:
            raise ValueError(f"unknown action {action!r}")
        return self._score_with_action(X, action)


def build_estimator(learner_type: str, *, base: str = "hist_gb") -> BaseUpliftEstimator:
    if learner_type == "t_learner":
        return TLearner(base=base)
    if learner_type == "s_learner":
        return SLearner(base=base)
    raise ValueError(f"unknown learner_type {learner_type!r}")
