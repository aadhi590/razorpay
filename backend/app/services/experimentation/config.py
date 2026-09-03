"""Experiment configuration.

A single, extensible value object (:class:`ExperimentConfig`) describing how the
action-assignment layer should behave. It is intentionally decoupled from
:mod:`app.config` (the app's ``BaseSettings``) so that enabling an experiment
never risks the app's core settings parsing; configuration is loaded explicitly
via :func:`load_experiment_config` (environment) or constructed directly (tests,
API query parameters).

Extensibility: :data:`KNOWN_STRATEGIES` currently covers ``exploit`` (no
randomization -- the default), ``epsilon_greedy`` and ``uniform``. A future
contextual-bandit policy is added by defining a new strategy name, a draw
function in :mod:`app.services.experimentation.assignment`, and (if needed) new
optional fields here -- without touching the orchestrator.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from app.services.recovery_config import ACTION_TYPES

# --- strategies ---------------------------------------------------------
STRATEGY_EXPLOIT = "exploit"
STRATEGY_EPSILON_GREEDY = "epsilon_greedy"
STRATEGY_UNIFORM = "uniform"
KNOWN_STRATEGIES = frozenset(
    {STRATEGY_EXPLOIT, STRATEGY_EPSILON_GREEDY, STRATEGY_UNIFORM}
)

# --- assignment-coverage / positivity thresholds (used by analytics) ---
# An action observed on fewer distinct events than this is flagged as having
# insufficient representation for action-level estimation.
MIN_DISTINCT_EVENTS_PER_ACTION = 30
# The smallest assignment propensity we consider safe for inverse-propensity
# weighting; below this, weights explode and estimates become unstable.
MIN_PROPENSITY_FOR_OVERLAP = 0.05

_ENV_PREFIX = "EXPERIMENTATION_"


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<MISSING>"


_MISSING = _Missing()


@dataclass(frozen=True)
class ExperimentConfig:
    """How to assign the executed action for treatment recovery events.

    Attributes
    ----------
    experiment_id:
        Logical identifier for the experiment this assignment belongs to
        (free-form string, not a DB foreign key). Recorded on every assignment.
    enabled:
        Master switch. When ``False`` the effective strategy is always
        ``exploit`` regardless of the other fields, so behaviour is identical
        to "no experimentation layer".
    strategy:
        One of :data:`KNOWN_STRATEGIES`.
    epsilon:
        Exploration probability for ``epsilon_greedy`` (in ``[0, 1]``). With
        probability ``1 - epsilon`` the top-ranked eligible action is taken;
        with probability ``epsilon`` a *different* eligible action is chosen
        uniformly at random.
    allowed_actions:
        If set, restricts the eligible set to (policy candidates) INTERSECT
        (allowed_actions). ``None`` means "all policy candidates are eligible".
        Must be a subset of the known action types.
    seed:
        Deterministic base seed for the assignment RNG. The per-decision RNG is
        derived as ``Random(f"{seed}:{recovery_event_id}")`` so runs are
        reproducible and independent of event ordering. ``None`` means use
        system entropy (only reached when ``enabled`` and exploring).
    variant:
        Variant label recorded on treatment assignments (control events never
        reach this layer).
    """

    experiment_id: str | None = None
    enabled: bool = False
    strategy: str = STRATEGY_EXPLOIT
    epsilon: float = 0.0
    allowed_actions: tuple[str, ...] | None = None
    seed: int | None = None
    variant: str = "treatment"

    def __post_init__(self) -> None:
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; "
                f"expected one of {sorted(KNOWN_STRATEGIES)}"
            )
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if self.allowed_actions is not None:
            unknown = [a for a in self.allowed_actions if a not in ACTION_TYPES]
            if unknown:
                raise ValueError(
                    f"allowed_actions contains unknown action types: {unknown}"
                )
            if len(self.allowed_actions) == 0:
                raise ValueError("allowed_actions must be non-empty when set")

    @property
    def effective_strategy(self) -> str:
        """The strategy actually applied. Disabled experiments always exploit."""
        if not self.enabled:
            return STRATEGY_EXPLOIT
        return self.strategy

    @property
    def explores(self) -> bool:
        """True if this config can ever assign a non-top-ranked action."""
        s = self.effective_strategy
        if s == STRATEGY_UNIFORM:
            return True
        if s == STRATEGY_EPSILON_GREEDY:
            return self.epsilon > 0.0
        return False

    def summary(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "enabled": self.enabled,
            "strategy": self.strategy,
            "effective_strategy": self.effective_strategy,
            "epsilon": self.epsilon,
            "allowed_actions": (
                list(self.allowed_actions)
                if self.allowed_actions is not None
                else None
            ),
            "seed": self.seed,
            "variant": self.variant,
        }


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def _env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_experiment_config(
    overrides: Mapping[str, object] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from environment variables, with an
    optional ``overrides`` mapping taking precedence (used for per-request API
    overrides).

    Environment variables (all optional)::

        EXPERIMENTATION_ENABLED         -> enabled           (bool)
        EXPERIMENTATION_STRATEGY        -> strategy           (str)
        EXPERIMENTATION_EPSILON         -> epsilon            (float)
        EXPERIMENTATION_SEED            -> seed               (int)
        EXPERIMENTATION_ALLOWED_ACTIONS -> allowed_actions    (comma-separated)
        EXPERIMENTATION_EXPERIMENT_ID   -> experiment_id      (str)
        EXPERIMENTATION_VARIANT         -> variant            (str)
    """
    env = env if env is not None else os.environ
    overrides = dict(overrides or {})

    def pick(key: str, env_key: str, cast):
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        raw = env.get(_ENV_PREFIX + env_key)
        if raw is None or raw == "":
            return _MISSING
        return cast(raw)

    kwargs: dict[str, object] = {}

    enabled = pick("enabled", "ENABLED", _env_bool)
    if enabled is not _MISSING:
        kwargs["enabled"] = bool(enabled)

    strategy = pick("strategy", "STRATEGY", lambda s: str(s).strip().lower())
    if strategy is not _MISSING:
        kwargs["strategy"] = strategy

    epsilon = pick("epsilon", "EPSILON", float)
    if epsilon is not _MISSING:
        kwargs["epsilon"] = epsilon

    seed = pick("seed", "SEED", int)
    if seed is not _MISSING:
        kwargs["seed"] = seed

    allowed = pick(
        "allowed_actions",
        "ALLOWED_ACTIONS",
        lambda s: tuple(p.strip() for p in str(s).split(",") if p.strip()),
    )
    if allowed is not _MISSING:
        kwargs["allowed_actions"] = (
            tuple(allowed) if allowed else None
        )

    experiment_id = pick("experiment_id", "EXPERIMENT_ID", str)
    if experiment_id is not _MISSING:
        kwargs["experiment_id"] = experiment_id

    variant = pick("variant", "VARIANT", str)
    if variant is not _MISSING:
        kwargs["variant"] = variant

    # epsilon > 0 with no explicit strategy implies epsilon-greedy.
    if (
        "strategy" not in kwargs
        and kwargs.get("epsilon", 0.0)
        and float(kwargs.get("epsilon", 0.0)) > 0.0
    ):
        kwargs["strategy"] = STRATEGY_EPSILON_GREEDY

    return ExperimentConfig(**kwargs)  # type: ignore[arg-type]
