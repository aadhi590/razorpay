"""Historical-incrementality tool.

``get_action_scores`` gives the models' **predicted** numbers for *this* event.
This tool gives the **observed, measured** incremental recovery lift an action
type has actually produced across every past recovery -- the real control-vs-
treated contrast, grouped by action.

It is a second, honest signal the agent may consult and reason over; it does not
retrain anything, does not change the ML/uplift models, and -- like every other
read tool -- cannot influence or bypass a guardrail. The existing guardrail layer
remains the sole authority on what the agent is actually allowed to execute.

The numbers come from :meth:`app.services.analytics_service.AnalyticsService.
action_incrementality`, which reuses the batch ``/analytics/recovery-impact``
endpoint's control-group definition (``RecoveryEvent.is_control``), its
recovered definition, its incremental-rate subtraction, and its Newcombe/Wilson
confidence interval -- not a second, different implementation.
"""
from __future__ import annotations

from typing import Any

from app.agent.tools.base import Tool, ToolContext, obj
from app.services.recovery_config import ACTION_TYPES

_SUPPORTED = sorted(ACTION_TYPES)


class GetHistoricalIncrementalityForAction(Tool):
    name = "get_historical_incrementality_for_action"
    description = (
        "REAL, measured historical performance of one recovery action type: the "
        "recovery rate actually observed across every past event where that "
        "action was executed, minus the randomized control baseline -- i.e. the "
        "incremental lift the action has genuinely delivered, with a 95% "
        "confidence interval. Use it to sanity-check a PREDICTED uplift from "
        "get_action_scores before committing to an action. All numbers are "
        "application-computed from historical outcomes; do not invent or change "
        "them."
    )
    parameters = obj(
        {
            "action_type": {
                "type": "string",
                "enum": _SUPPORTED,
                "description": "The recovery action type to look up.",
            },
        },
        required=["action_type"],
    )
    output_schema = obj({
        "action_type": {"type": "string"},
        "computable": {"type": "boolean"},
        "reason": {"type": "string"},
        "treated_group_size": {"type": "integer"},
        "control_group_size": {"type": "integer"},
        "observed_recovery_rate_for_action": {"type": "number"},
        "baseline_control_recovery_rate": {"type": "number"},
        "observed_incremental_lift": {"type": "number"},
        "observed_incremental_lift_ci_95": {"type": "array"},
        "model_predicted_uplift_for_context": {"type": "number"},
        "note": {"type": "string"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        action_type = str(args.get("action_type", ""))
        # Defensive re-check (the loop's arg validator already enforces the enum;
        # this covers a direct call and keeps the failure shape consistent).
        if action_type not in ACTION_TYPES:
            return {
                "error": "unknown_action_type",
                "message": f"{action_type!r} is not a supported action",
                "supported": _SUPPORTED,
            }

        from app.services.analytics_service import AnalyticsService

        try:
            r = AnalyticsService(ctx.db).action_incrementality(action_type)
        except Exception as exc:  # noqa: BLE001 - a query failure must not crash the run
            ctx.state.errors.append(
                f"historical incrementality lookup failed: {type(exc).__name__}"
            )
            return {
                "action_type": action_type,
                "computable": False,
                "reason": "lookup_failed",
                "note": "the historical incrementality query could not be completed",
            }

        payload: dict[str, Any] = {
            "action_type": r.action_type,
            "computable": r.computable,
            "treated_group_size": r.treated_group_size,
            "control_group_size": r.control_group_size,
            "note": r.sample_size_note,
        }
        if not r.computable:
            payload["reason"] = r.reason
        else:
            payload.update(
                {
                    "observed_recovery_rate_for_action": r.observed_recovery_rate_for_action,
                    "baseline_control_recovery_rate": r.baseline_control_recovery_rate,
                    "observed_incremental_lift": r.observed_incremental_lift,
                    "observed_incremental_lift_ci_95": r.observed_incremental_lift_ci_95,
                }
            )
            predicted = self._predicted_uplift(ctx, action_type)
            if predicted is not None:
                payload["model_predicted_uplift_for_context"] = predicted

        # Accumulate on the state so the persisted trace records exactly what the
        # agent looked up (one entry per action queried) -- same pattern as
        # get_action_scores -> state.quantitative_scores.
        store = dict(ctx.state.action_incrementality or {})
        store[action_type] = payload
        ctx.state.action_incrementality = store
        return payload

    @staticmethod
    def _predicted_uplift(ctx: ToolContext, action_type: str) -> float | None:
        """The uplift the uplift model predicted for THIS event's context, if
        get_action_scores was already called this run. Read straight from the
        state slot that tool populated -- no re-inference, no duplicated logic.
        Omitted entirely when scores were not fetched."""
        scores = ctx.state.quantitative_scores or []
        for s in scores:
            if s.get("action") == action_type and s.get("uplift") is not None:
                return round(float(s["uplift"]), 4)
        return None


INSIGHT_TOOLS = [GetHistoricalIncrementalityForAction()]
