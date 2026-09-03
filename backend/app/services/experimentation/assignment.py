"""Action assignment: decide which *eligible* action is actually executed, and
record the exact probability with which it was chosen (its propensity).

Separation of concerns
----------------------
* The policy (:class:`~app.services.recovery_policy.RecoveryPolicy`) produces a
  ranked list of eligible :class:`~app.services.recovery_policy.CandidateAction`
  objects. It already applies all eligibility filtering (already-tried actions,
  premium gating, control -> no candidates).
* :class:`ActionAssigner` takes that ``PolicyDecision`` plus an
  :class:`~app.services.experimentation.config.ExperimentConfig` and returns an
  :class:`AssignmentResult`: the chosen action, its propensity, and the full
  audit trail of the assignment mechanism.

The assigner NEVER widens the eligible set -- exploration happens strictly
inside ``decision.candidates`` (optionally further narrowed by
``config.allowed_actions``). It never proposes an action the policy did not.

Propensity
----------
``propensity`` = P(chosen action | observed context, assignment mechanism),
*after* eligibility filtering. For epsilon-greedy over ``k`` eligible actions
ranked ``a_1`` (top) .. ``a_k``::

    P(a_1) = 1 - epsilon                 (only the exploit branch picks the top)
    P(a_i) = epsilon / (k - 1)   for i >= 2

With ``k == 1`` or ``epsilon == 0`` the top action is taken with propensity 1.0.
For the ``uniform`` strategy every eligible action has propensity ``1 / k``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from app.services.experimentation.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    STRATEGY_EPSILON_GREEDY,
    STRATEGY_EXPLOIT,
    STRATEGY_UNIFORM,
    ExperimentConfig,
)
from app.services.recovery_policy import PolicyDecision


@dataclass(frozen=True)
class AssignmentResult:
    """Everything needed to audit one action assignment and to later use it for
    inverse-propensity / doubly-robust estimation."""

    chosen_action: str
    propensity: float
    exploration: bool
    # eligible set the assignment ranged over, in policy-rank order (best-first)
    eligible_actions: list[str]
    # the policy's full ranking (best-first), before allowed_actions narrowing
    policy_ranking: list[str]
    strategy: str
    epsilon: float
    assignment_mechanism: str
    experiment_id: str | None
    variant: str
    policy_name: str
    model_version: str | None
    rng_seed: int | None
    # notes: e.g. allowed_actions removed every policy candidate -> fell back
    notes: list[str] = field(default_factory=list)

    def as_context_dict(self) -> dict:
        """Shape persisted into ``AgentEvent.input_context['assignment']``."""
        return {
            "chosen_action": self.chosen_action,
            "propensity": self.propensity,
            "exploration": self.exploration,
            "eligible_actions": list(self.eligible_actions),
            "policy_ranking": list(self.policy_ranking),
            "strategy": self.strategy,
            "epsilon": self.epsilon,
            "assignment_mechanism": self.assignment_mechanism,
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "policy_name": self.policy_name,
            "model_version": self.model_version,
            "rng_seed": self.rng_seed,
            "notes": list(self.notes),
        }


def _draw(
    strategy: str,
    eligible: list[str],
    epsilon: float,
    rng: random.Random,
) -> tuple[str, float, bool]:
    """Return ``(chosen_action, propensity, exploration)``."""
    k = len(eligible)
    top = eligible[0]

    if strategy == STRATEGY_EXPLOIT or k == 1:
        return top, 1.0, False

    if strategy == STRATEGY_UNIFORM:
        choice = rng.choice(eligible)
        return choice, 1.0 / k, choice != top

    # epsilon-greedy
    if epsilon <= 0.0:
        return top, 1.0, False
    if rng.random() < epsilon:
        others = eligible[1:]
        choice = rng.choice(others)
        return choice, epsilon / (k - 1), True
    return top, 1.0 - epsilon, False


def _mechanism_label(strategy: str, k: int, epsilon: float) -> str:
    if strategy == STRATEGY_EPSILON_GREEDY:
        return f"epsilon_greedy(k={k}, epsilon={epsilon})"
    if strategy == STRATEGY_UNIFORM:
        return f"uniform_random(k={k})"
    return f"exploit(k={k})"


class ActionAssigner:
    """Assigns the executed action for a treatment recovery event.

    With :data:`DEFAULT_EXPERIMENT_CONFIG` (disabled) this is a pass-through:
    the top-ranked candidate is always chosen with ``propensity = 1.0`` and
    ``exploration = False`` -- byte-for-byte the orchestrator's prior behaviour.

    Determinism: pass an explicit ``rng`` for tests, or set ``config.seed`` so
    the per-decision RNG is ``Random(f"{seed}:{recovery_event_id}")``.
    """

    def __init__(
        self,
        config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self._rng = rng

    def _rng_for(self, recovery_event_id: int) -> random.Random:
        if self._rng is not None:
            return self._rng
        if self.config.seed is not None:
            return random.Random(f"{self.config.seed}:{recovery_event_id}")
        return random.Random()

    def assign(
        self,
        *,
        decision: PolicyDecision,
        recovery_event_id: int,
        policy_name: str,
        model_version: str | None = None,
    ) -> AssignmentResult:
        cfg = self.config
        policy_ranking = [c.action_type for c in decision.candidates]
        if not policy_ranking:
            raise ValueError(
                "ActionAssigner.assign called with no policy candidates; "
                "the orchestrator must handle an empty decision before assigning"
            )

        notes: list[str] = []
        if cfg.allowed_actions is not None:
            allowed = set(cfg.allowed_actions)
            eligible = [a for a in policy_ranking if a in allowed]
            if not eligible:
                eligible = list(policy_ranking)
                notes.append(
                    "allowed_actions removed every policy candidate; "
                    "fell back to the full policy ranking with no exploration"
                )
        else:
            eligible = list(policy_ranking)

        strategy = cfg.effective_strategy
        if notes and "fell back" in notes[-1]:
            strategy = STRATEGY_EXPLOIT

        rng = self._rng_for(recovery_event_id)
        chosen, propensity, exploration = _draw(
            strategy, eligible, cfg.epsilon, rng
        )

        return AssignmentResult(
            chosen_action=chosen,
            propensity=round(float(propensity), 8),
            exploration=exploration,
            eligible_actions=eligible,
            policy_ranking=policy_ranking,
            strategy=strategy,
            epsilon=cfg.epsilon if strategy == STRATEGY_EPSILON_GREEDY else 0.0,
            assignment_mechanism=_mechanism_label(
                strategy, len(eligible), cfg.epsilon
            ),
            experiment_id=cfg.experiment_id,
            variant=cfg.variant,
            policy_name=policy_name,
            model_version=model_version,
            rng_seed=cfg.seed,
            notes=notes,
        )
