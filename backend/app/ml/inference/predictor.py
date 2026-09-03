"""Inference: load an artifact and score ``P(recovery | context, action)``.

``RecoveryModel`` is the only thing the serving path (``MLRecoveryPolicy``, the
read-only ML API) touches. It:

* loads the artifact once (``RecoveryModel.load``), raising ``ModelUnavailable``
  if it is missing / corrupt / schema-mismatched -- callers fall back;
* applies the EXACT preprocessing baked into the fitted pipeline;
* validates inputs (known action, required feature columns);
* returns probabilities, never bare class labels;
* offers a lightweight ``explain`` using the artifact's stored interpretability.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Connection

from app.ml.config import ACTION_COST_PAISE, ACTIONS
from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.models.artifact import ModelArtifact, ModelUnavailable, load_artifact


@dataclass(frozen=True)
class ActionScore:
    action: str
    probability: float
    cost_paise: int
    expected_value_paise: float


class RecoveryModel:
    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact
        self._features = artifact.all_features
        self._extractor = PointInTimeFeatureExtractor()

    # -- construction ------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "RecoveryModel":
        return cls(load_artifact(path))

    @classmethod
    def try_load(cls, path: Path | None = None) -> "RecoveryModel | None":
        try:
            return cls.load(path)
        except ModelUnavailable:
            return None

    # -- identity --------------------------------------------------
    @property
    def version(self) -> str:
        return self._artifact.model_version

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    # -- core scoring --------------------------------------------
    def predict_from_frame(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self._features if c not in df.columns]
        if missing:
            raise ValueError(f"feature frame is missing columns: {missing}")
        X = df[self._features]
        proba = self._artifact.estimator.predict_proba(X)
        return np.asarray(proba)[:, 1]

    def predict_recovery(self, features: dict, action: str) -> float:
        self._validate_action(action)
        row = dict(features)
        row["action"] = action
        df = pd.DataFrame([row])
        return float(self.predict_from_frame(df)[0])

    def predict_all_actions(
        self, features: dict, actions: list[str] | None = None
    ) -> dict[str, float]:
        actions = actions or list(ACTIONS)
        for a in actions:
            self._validate_action(a)
        rows = [{**features, "action": a} for a in actions]
        probs = self.predict_from_frame(pd.DataFrame(rows))
        return {a: float(p) for a, p in zip(actions, probs)}

    # -- DB-backed scoring (train == inference feature path) --------
    def predict_for_event(
        self,
        conn: Connection,
        recovery_event_id: int,
        *,
        actions: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, ActionScore]:
        as_of = as_of or datetime.now(timezone.utc)
        feat = self._extractor.features_for_event(conn, recovery_event_id, as_of)
        wanted = set(actions) if actions else set(ACTIONS)
        feat = feat[feat["action"].isin(wanted)].reset_index(drop=True)
        if feat.empty:
            raise ValueError(f"no requested actions among {sorted(wanted)}")
        probs = self.predict_from_frame(feat)

        amount = float(feat["amount_paise"].iloc[0])
        scores: dict[str, ActionScore] = {}
        for action, p in zip(feat["action"].tolist(), probs):
            cost = ACTION_COST_PAISE.get(action, 0)
            scores[action] = ActionScore(
                action=action,
                probability=float(p),
                cost_paise=cost,
                expected_value_paise=round(float(p) * amount - cost, 2),
            )
        return scores

    # -- explanation --------------------------------------------
    def explain(self, features: dict, action: str, top_k: int = 6) -> dict:
        """Best-effort explanation from the artifact's stored interpretability.

        Uses the standalone logistic-regression coefficient table (sign =
        direction of effect) intersected with this row's feature values, plus
        the selected model's global permutation importances. Not a per-sample
        SHAP attribution.
        """
        self._validate_action(action)
        row = {**features, "action": action}
        interp = self._artifact.interpretability
        coefs = {r["feature"]: r["coefficient"]
                 for r in interp.get("logistic_regression_coefficients", [])}
        perm = {r["feature"]: r["importance_mean"]
                for r in interp.get("permutation_importance_test", [])}

        contributions = []
        for raw_feat in self._artifact.numeric_features:
            val = row.get(raw_feat)
            if val is None:
                continue
            # match one-hot / scaled names loosely by prefix
            c = coefs.get(f"num__{raw_feat}")
            if c is None:
                continue
            contributions.append({
                "feature": raw_feat,
                "value": val,
                "logreg_coefficient": c,
                "direction": "increases" if c > 0 else "decreases",
            })
        for cat_feat in self._artifact.categorical_features:
            val = row.get(cat_feat)
            if val is None:
                continue
            key = f"cat__{cat_feat}_{val}"
            c = coefs.get(key)
            if c is None:
                continue
            contributions.append({
                "feature": f"{cat_feat}={val}",
                "value": val,
                "logreg_coefficient": c,
                "direction": "increases" if c > 0 else "decreases",
            })
        contributions.sort(key=lambda d: -abs(d["logreg_coefficient"]))

        return {
            "action": action,
            "predicted_probability": None,  # filled by caller if desired
            "top_factors": contributions[:top_k],
            "model_global_importance_top": sorted(
                perm.items(), key=lambda kv: -kv[1]
            )[:top_k],
            "note": interp.get("note", ""),
        }

    # -- helpers -----------------------------------------------
    def _validate_action(self, action: str) -> None:
        if action not in ACTIONS:
            raise ValueError(
                f"unknown action {action!r}; known: {list(ACTIONS)}"
            )


@lru_cache(maxsize=1)
def get_default_model() -> RecoveryModel | None:
    """Process-wide cached handle to the latest artifact (or ``None``)."""
    return RecoveryModel.try_load()


def reset_default_model() -> None:
    get_default_model.cache_clear()
