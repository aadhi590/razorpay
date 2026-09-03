"""Uplift inference: baseline + per-action uplift + incremental economic value.

``UpliftModel`` is the only thing the serving path (``UpliftRecoveryPolicy``,
the read-only uplift API) touches. It:

* loads the uplift artifact once, raising ``UpliftModelUnavailable`` when it is
  missing / corrupt / schema-mismatched -- callers fall back;
* reuses the predictive layer's point-in-time feature extractor, so an uplift
  score and a predictive score for the same event see identical features;
* returns calibrated probabilities and their difference (uplift), never bare
  labels;
* converts uplift into money:
      incremental_expected_revenue = uplift * payment_amount_paise
      net_incremental_value        = incremental_expected_revenue - action_cost
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Connection

from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.uplift.config import INTERVENTION_COST_PAISE, TREATMENT_ACTIONS
from app.ml.uplift.models.artifact import (
    UpliftArtifact,
    UpliftModelUnavailable,
    load_uplift_artifact,
)


@dataclass(frozen=True)
class ActionUplift:
    action: str
    treatment_probability: float
    uplift: float
    cost_paise: int
    incremental_expected_revenue_paise: float
    net_incremental_value_paise: float
    rank: int


@dataclass(frozen=True)
class EventUplift:
    recovery_event_id: int
    baseline_probability: float
    amount_paise: int
    actions: list[ActionUplift]
    recommended_action: str | None
    model_version: str

    def as_dict(self) -> dict:
        return {
            "recovery_event_id": self.recovery_event_id,
            "baseline_probability": round(self.baseline_probability, 6),
            "amount_paise": self.amount_paise,
            "actions": [
                {
                    "action": a.action,
                    "treatment_probability": round(a.treatment_probability, 6),
                    "uplift": round(a.uplift, 6),
                    "cost_paise": a.cost_paise,
                    "incremental_expected_revenue_paise": round(
                        a.incremental_expected_revenue_paise, 2
                    ),
                    "net_incremental_value_paise": round(
                        a.net_incremental_value_paise, 2
                    ),
                    "rank": a.rank,
                }
                for a in self.actions
            ],
            "recommended_action": self.recommended_action,
            "model_version": self.model_version,
        }


class UpliftModel:
    def __init__(self, artifact: UpliftArtifact) -> None:
        self._artifact = artifact
        self._estimator = artifact.estimator
        self._extractor = PointInTimeFeatureExtractor()
        self._actions = list(artifact.treatment_actions or TREATMENT_ACTIONS)

    @classmethod
    def load(cls, path: Path | None = None) -> "UpliftModel":
        return cls(load_uplift_artifact(path))

    @classmethod
    def try_load(cls, path: Path | None = None) -> "UpliftModel | None":
        try:
            return cls.load(path)
        except UpliftModelUnavailable:
            return None

    @property
    def version(self) -> str:
        return self._artifact.model_version

    @property
    def artifact(self) -> UpliftArtifact:
        return self._artifact

    # -- scoring from a prepared context frame (1 row) ----------
    def score_context(
        self,
        context_row: pd.DataFrame,
        *,
        amount_paise: int,
        actions: list[str] | None = None,
    ) -> tuple[float, list[ActionUplift]]:
        if len(context_row) != 1:
            raise ValueError("score_context expects exactly one context row")
        actions = [a for a in (actions or self._actions) if a in self._actions]
        baseline = float(self._estimator.predict_baseline(context_row)[0])

        scored: list[ActionUplift] = []
        for a in actions:
            p_a = float(self._estimator.predict_action(context_row, a)[0])
            uplift = p_a - baseline
            cost = INTERVENTION_COST_PAISE.get(a, 0)
            inc_rev = uplift * float(amount_paise)
            scored.append(
                ActionUplift(
                    action=a,
                    treatment_probability=p_a,
                    uplift=uplift,
                    cost_paise=cost,
                    incremental_expected_revenue_paise=inc_rev,
                    net_incremental_value_paise=inc_rev - cost,
                    rank=0,
                )
            )
        scored.sort(key=lambda s: (-s.net_incremental_value_paise, s.cost_paise))
        scored = [
            ActionUplift(**{**s.__dict__, "rank": i + 1})
            for i, s in enumerate(scored)
        ]
        return baseline, scored

    # -- DB-backed scoring (train == inference feature path) ----
    def predict_for_event(
        self,
        conn: Connection,
        recovery_event_id: int,
        *,
        actions: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> EventUplift:
        as_of = as_of or datetime.now(timezone.utc)
        feat = self._extractor.features_for_event(conn, recovery_event_id, as_of)
        context = feat.iloc[[0]].reset_index(drop=True)
        amount_paise = int(context["amount_paise"].iloc[0])

        baseline, scored = self.score_context(
            context, amount_paise=amount_paise, actions=actions
        )
        positive = [s for s in scored if s.net_incremental_value_paise > 0]
        recommended = positive[0].action if positive else None
        return EventUplift(
            recovery_event_id=recovery_event_id,
            baseline_probability=baseline,
            amount_paise=amount_paise,
            actions=scored,
            recommended_action=recommended,
            model_version=self.version,
        )
