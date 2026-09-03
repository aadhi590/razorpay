"""Action-assignment / experimentation layer.

This package is deliberately separate from :mod:`app.services.recovery_policy`
and :mod:`app.services.ml_recovery_policy`. Those *rank* candidate actions by
predicted recovery probability / expected value. This package *assigns* the
action that is actually executed, and records the exact probability with which
that action was assigned (its **propensity**) so the resulting data can later
support action-level causal / uplift estimation.

Nothing here changes the default behaviour of the orchestrator: the default
:data:`DEFAULT_EXPERIMENT_CONFIG` is disabled (``enabled=False``,
``epsilon=0``), which makes :class:`ActionAssigner` a pure pass-through that
always exploits the top-ranked candidate with ``propensity = 1.0``.
"""
from __future__ import annotations

from app.services.experimentation.assignment import (
    ActionAssigner,
    AssignmentResult,
)
from app.services.experimentation.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    KNOWN_STRATEGIES,
    MIN_DISTINCT_EVENTS_PER_ACTION,
    MIN_PROPENSITY_FOR_OVERLAP,
    STRATEGY_EPSILON_GREEDY,
    STRATEGY_EXPLOIT,
    STRATEGY_UNIFORM,
    ExperimentConfig,
    load_experiment_config,
)

__all__ = [
    "ActionAssigner",
    "AssignmentResult",
    "ExperimentConfig",
    "DEFAULT_EXPERIMENT_CONFIG",
    "load_experiment_config",
    "KNOWN_STRATEGIES",
    "STRATEGY_EXPLOIT",
    "STRATEGY_EPSILON_GREEDY",
    "STRATEGY_UNIFORM",
    "MIN_DISTINCT_EVENTS_PER_ACTION",
    "MIN_PROPENSITY_FOR_OVERLAP",
]
