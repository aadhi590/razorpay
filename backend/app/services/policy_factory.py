"""Resolve a recovery policy by name for the API layer.

Default is always ``rules`` (the deterministic, dependency-free baseline).
``ml`` returns :class:`MLRecoveryPolicy`, which itself falls back to the rules
policy if the model artifact is unavailable -- so ``?policy=ml`` can never break
the recovery workflow.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.experimentation import (
    STRATEGY_EPSILON_GREEDY,
    ActionAssigner,
    load_experiment_config,
)
from app.services.recovery_policy import RecoveryPolicy, RulesBasedRecoveryPolicy

POLICY_NAMES = ("rules", "ml", "uplift")
DEFAULT_POLICY = "rules"


def resolve_policy(name: str | None, db: Session) -> RecoveryPolicy:
    key = (name or DEFAULT_POLICY).lower()
    if key in ("rules", "rules_v1", "default"):
        return RulesBasedRecoveryPolicy()
    if key in ("ml", "ml_v1", "model"):
        # imported lazily so the API has no hard dependency on the ML stack
        from app.services.ml_recovery_policy import MLRecoveryPolicy

        return MLRecoveryPolicy(db)
    if key in ("uplift", "causal", "uplift_v1"):
        # uplift -> ML -> rules fallback chain, all lazily imported
        from app.services.uplift_recovery_policy import UpliftRecoveryPolicy

        return UpliftRecoveryPolicy(db)
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")


def resolve_assigner(
    *,
    epsilon: float | None = None,
    experiment_id: str | None = None,
    seed: int | None = None,
) -> ActionAssigner:
    """Build the action assigner for an orchestrator run.

    Base configuration comes from the ``EXPERIMENTATION_*`` environment
    variables (see :func:`app.services.experimentation.load_experiment_config`).
    A request may override it: passing ``epsilon > 0`` turns on epsilon-greedy
    exploration for that run only. With no env config and no override this is
    the disabled/exploit assigner -- identical to no experimentation layer.
    """
    overrides: dict[str, object] = {}
    if epsilon is not None and epsilon > 0.0:
        overrides["enabled"] = True
        overrides["epsilon"] = epsilon
        overrides["strategy"] = STRATEGY_EPSILON_GREEDY
    if experiment_id:
        overrides["experiment_id"] = experiment_id
    if seed is not None:
        overrides["seed"] = seed
    return ActionAssigner(load_experiment_config(overrides))
