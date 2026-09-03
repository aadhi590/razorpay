"""Quantitative-scoring tool.

This is the ONLY way Gemini gets numbers. Every probability, uplift and
expected-value figure originates in the existing quantitative layer
(:class:`~app.ml.inference.predictor.RecoveryModel` and
:class:`~app.ml.uplift.inference.predictor.UpliftModel`). If neither artifact is
available the tool falls back to the rules policy's expected-value arithmetic
(still application-computed) and says so. Gemini may reason over and choose
among these numbers; it may not invent or modify them.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agent.guardrails import (
    build_policy_context,
    eligible_action_types,
    load_event,
)
from app.agent.tools.base import Tool, ToolContext, obj
from app.services.recovery_policy import RulesBasedRecoveryPolicy


class GetActionScores(Tool):
    name = "get_action_scores"
    description = (
        "Quantitative scores for the currently-eligible recovery actions: "
        "calibrated recovery probability and expected value from the ML model, "
        "plus causal uplift and net incremental value from the uplift model. "
        "All numbers come from the models; do not invent or change them."
    )
    parameters = obj({})
    output_schema = obj({
        "as_of": {"type": "string"},
        "source": {"type": "string"},
        "ml_model_version": {"type": "string"},
        "uplift_model_version": {"type": "string"},
        "scores": {
            "type": "array",
            "items": obj({
                "action": {"type": "string"},
                "recovery_probability": {"type": "number"},
                "expected_value_paise": {"type": "number"},
                "uplift": {"type": "number"},
                "net_incremental_value_paise": {"type": "number"},
                "cost_paise": {"type": "integer"},
                "eligible": {"type": "boolean"},
            }),
        },
        "recommended_by_expected_value": {"type": "string"},
        "recommended_by_uplift": {"type": "string"},
        "note": {"type": "string"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event = load_event(ctx.db, ctx.state.recovery_event_id)
        if event is None:
            return {"error": "event_not_found",
                    "recovery_event_id": ctx.state.recovery_event_id}
        if event.is_control:
            return {"scores": [], "note": "control event: no actions are scored"}

        eligible = eligible_action_types(event)
        if not eligible:
            return {"scores": [], "note": "no eligible untried actions to score"}

        as_of = datetime.now(timezone.utc)
        ml_scores = self._ml_scores(ctx, event, eligible, as_of)
        uplift_scores = self._uplift_scores(ctx, event, eligible, as_of)

        source_parts = []
        if ml_scores is not None:
            source_parts.append("ml")
        if uplift_scores is not None:
            source_parts.append("uplift")
        used_rules_fallback = not source_parts

        rules_ev = self._rules_ev(event, eligible) if used_rules_fallback else {}

        scores = []
        for a in eligible:
            ml = (ml_scores or {}).get(a)
            up = (uplift_scores or {}).get(a)
            row: dict[str, Any] = {
                "action": a,
                "eligible": True,
                "cost_paise": (
                    ml["cost_paise"] if ml else
                    up["cost_paise"] if up else
                    rules_ev.get(a, {}).get("cost_paise")
                ),
            }
            if ml:
                row["recovery_probability"] = round(ml["probability"], 4)
                row["expected_value_paise"] = round(ml["expected_value_paise"], 2)
            if up:
                row["uplift"] = round(up["uplift"], 4)
                row["treatment_probability"] = round(up["treatment_probability"], 4)
                row["net_incremental_value_paise"] = round(
                    up["net_incremental_value_paise"], 2
                )
            if used_rules_fallback and a in rules_ev:
                row["recovery_probability"] = rules_ev[a]["probability"]
                row["expected_value_paise"] = rules_ev[a]["expected_value_paise"]
            scores.append(row)

        rec_ev = self._argmax(scores, "expected_value_paise")
        rec_uplift = self._argmax(scores, "net_incremental_value_paise")

        payload: dict[str, Any] = {
            "as_of": as_of.isoformat(),
            "source": "+".join(source_parts) if source_parts else "rules_fallback",
            "scores": scores,
            "recommended_by_expected_value": rec_ev,
            "recommended_by_uplift": rec_uplift,
            "note": (
                "SYNTHETIC BENCHMARK models. "
                + (
                    "ML and uplift artifacts unavailable; expected value computed "
                    "from the rules policy."
                    if used_rules_fallback
                    else "Values are calibrated model outputs; choose among eligible "
                    "actions, do not modify the numbers."
                )
            ),
        }
        ctx.state.quantitative_scores = scores
        return payload

    # -- helpers -----------------------------------------------------
    @staticmethod
    def _ml_scores(ctx, event, eligible, as_of):
        try:
            from app.ml.inference.predictor import RecoveryModel

            model = RecoveryModel.try_load()
            if model is None:
                return None
            scored = model.predict_for_event(
                ctx.db.connection(), event.id, actions=eligible, as_of=as_of
            )
            return {
                a: {
                    "probability": float(s.probability),
                    "cost_paise": int(s.cost_paise),
                    "expected_value_paise": float(s.expected_value_paise),
                }
                for a, s in scored.items()
            }
        except Exception as exc:  # noqa: BLE001 - tool must never raise into the loop
            ctx.state.errors.append(f"ml scoring failed: {type(exc).__name__}")
            return None

    @staticmethod
    def _uplift_scores(ctx, event, eligible, as_of):
        try:
            from app.ml.uplift.inference.predictor import UpliftModel

            model = UpliftModel.try_load()
            if model is None:
                return None
            result = model.predict_for_event(
                ctx.db.connection(), event.id, actions=eligible, as_of=as_of
            )
            return {
                a.action: {
                    "uplift": float(a.uplift),
                    "treatment_probability": float(a.treatment_probability),
                    "cost_paise": int(a.cost_paise),
                    "net_incremental_value_paise": float(a.net_incremental_value_paise),
                }
                for a in result.actions
            }
        except Exception as exc:  # noqa: BLE001
            ctx.state.errors.append(f"uplift scoring failed: {type(exc).__name__}")
            return None

    @staticmethod
    def _rules_ev(event, eligible):
        decision = RulesBasedRecoveryPolicy().decide(build_policy_context(event))
        return {
            c.action_type: {
                "probability": c.estimated_recovery_probability,
                "expected_value_paise": c.expected_value_paise,
                "cost_paise": c.cost_paise,
            }
            for c in decision.candidates
            if c.action_type in eligible
        }

    @staticmethod
    def _argmax(scores: list[dict], key: str) -> str | None:
        vals = [s for s in scores if s.get(key) is not None]
        if not vals:
            return None
        return max(vals, key=lambda s: s[key])["action"]


SCORE_TOOLS = [GetActionScores()]
