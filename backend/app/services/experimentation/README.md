# Experimentation / Action-Assignment Layer (`app/services/experimentation/`)

> **SYNTHETIC BENCHMARK.** Everything downstream of this layer is still trained
> and evaluated on data from `app/scripts/generate_data.py`. This layer changes
> *how the executed action is chosen and logged* so that a later causal / uplift
> model has the data it needs — it does **not** itself make anything causal.

## 1. Why this exists — ranking vs assignment

The recovery policies (`RulesBasedRecoveryPolicy`, `MLRecoveryPolicy`) answer
**"which actions are worth trying, and in what order?"** — they return a ranked
list of eligible `CandidateAction`s. Until now the orchestrator always executed
`candidates[0]` (pure exploitation).

Pure exploitation is fine for *serving* but useless for *learning about actions*:
the action taken is a deterministic function of the context, so action effects
are perfectly confounded with the context. You can never estimate
"what would `sms_nudge` have done on this event?" because `sms_nudge` is only
ever chosen for a specific, non-random slice of contexts.

This layer introduces a clean separation:

| Concern | Component | Question |
|---|---|---|
| **Scoring** | `RecoveryModel` | `P(recovery \| context, action)` |
| **Ranking** | `RecoveryPolicy` | eligible actions, best-first, by expected value |
| **Assignment** | `ActionAssigner` (this layer) | which eligible action is *actually executed*, and with what probability |
| **Persistence** | `RecoveryOrchestratorService` | write Intervention / AgentEvent / AuditLog |

Randomization logic lives **only** here. The policies are untouched.

## 2. Epsilon-greedy mechanism

Given the policy's ranked, eligible actions `a₁` (top) … `a_k`:

* with probability **`1 − ε`**: execute the top-ranked action `a₁` (exploit)
* with probability **`ε`**: execute a *different* eligible action, chosen
  uniformly at random from `a₂ … a_k` (explore)

`ε` is configurable; **the default is `ε = 0`**, which is identical to the
previous behaviour (always `a₁`, propensity 1.0, no exploration).

Other strategies are available through the same interface:
`exploit` (default, no randomization) and `uniform` (every eligible action with
probability `1/k` — used by the generator's `--randomized-assignment` mode).

## 3. Propensity — definition and formula

**`propensity` = P(chosen action | observed context, assignment mechanism),
evaluated *after* eligibility filtering.**

It is the probability with which *this* action was assigned on *this* decision,
under the mechanism that was actually in force. For epsilon-greedy over `k`
eligible actions:

```
P(a₁)  = 1 − ε                 # only the exploit branch can pick the top action
P(aᵢ)  = ε / (k − 1)   for i ≥ 2
```

Edge cases: `k == 1` or `ε == 0` ⇒ `propensity = 1.0`. For `uniform`,
`propensity = 1/k` for every eligible action.

### Example

Context: ₹999 payment, `bank_timeout`, attempt 1. The ML policy ranks the four
eligible actions:

```
1. method_switch_prompt   (top)
2. whatsapp_nudge
3. sms_nudge
4. retry
```

Run with `ε = 0.25`, `k = 4`:

```
P(method_switch_prompt) = 1 − 0.25            = 0.75
P(whatsapp_nudge)       = 0.25 / (4 − 1)      = 0.08333…
P(sms_nudge)            = 0.25 / 3            = 0.08333…
P(retry)                = 0.25 / 3            = 0.08333…
                                          sum = 1.0  ✓
```

If the RNG explores and lands on `sms_nudge`, the intervention is executed with
`sms_nudge` and the AgentEvent records `propensity = 0.08333333`,
`exploration = true`. If it exploits, `method_switch_prompt` is executed with
`propensity = 0.75`, `exploration = false`.

## 4. Why propensity logging is necessary

To estimate an action's effect from *observational* logged data you must
reweight each observation by the inverse of the probability it was assigned
(inverse-propensity weighting, IPW; or its doubly-robust variants). That is only
possible if the assignment probability was **recorded at decision time** —
it cannot be reconstructed afterwards, because it depends on the exact eligible
set, ranking, strategy and `ε` in force for that specific decision.

Logging the propensity now means every intervention collected from here on is a
valid IPW sample. Interventions with no logged propensity (all historical data,
and any run with the default `ε = 0`) are **not** usable for IPW and are flagged
as such by `GET /api/v1/analytics/assignment-coverage`.

## 5. What is stored, and where

No schema change. The assignment record is written to the existing
`AgentEvent.input_context` JSON column under the `assignment` key:

```jsonc
"assignment": {
  "chosen_action": "sms_nudge",
  "propensity": 0.08333333,
  "exploration": true,
  "eligible_actions": ["method_switch_prompt", "whatsapp_nudge", "sms_nudge", "retry"],
  "policy_ranking":   ["method_switch_prompt", "whatsapp_nudge", "sms_nudge", "retry"],
  "strategy": "epsilon_greedy",
  "epsilon": 0.25,
  "assignment_mechanism": "epsilon_greedy(k=4, epsilon=0.25)",
  "experiment_id": "q3_channel_test",
  "variant": "treatment",
  "policy_name": "ml:ml_v1",
  "model_version": "ml_v1",
  "rng_seed": 42,
  "notes": []
}
```

The same fields are echoed on the orchestrator response
(`propensity`, `exploration`, `assignment`) and summarised in the AuditLog
metadata.

**Why JSON and not dedicated columns:** assignment metadata is decision-audit
data, which is exactly what `input_context` is for; volume is one small object
per intervention; the coverage analytics query it fine with Postgres JSON
operators. Dedicated indexed columns (or an `assignment_log` table with a
`CHECK (propensity > 0)` constraint) become worthwhile once experiments run at
production scale and IPW queries need to be fast — that is called out as the
recommended next step, not done now.

## 6. Configuration

`ExperimentConfig` (see `config.py`) is the single, extensible config object:

| field | meaning | default |
|---|---|---|
| `experiment_id` | logical id recorded on every assignment | `None` |
| `enabled` | master switch; `False` ⇒ always `exploit` | `False` |
| `strategy` | `exploit` \| `epsilon_greedy` \| `uniform` | `exploit` |
| `epsilon` | exploration rate for `epsilon_greedy`, in `[0, 1]` | `0.0` |
| `allowed_actions` | restrict eligible set to (policy candidates ∩ this) | `None` (all) |
| `seed` | base seed; per-decision RNG = `Random(f"{seed}:{recovery_event_id}")` | `None` |
| `variant` | variant label on treatment assignments | `"treatment"` |

Loaded from `EXPERIMENTATION_*` environment variables via
`load_experiment_config()`, or overridden per request by the
`?epsilon=&experiment_id=&assignment_seed=` query params on
`POST /api/v1/orchestrator/run` and `.../orchestrate`.

Extending later (uniform randomization, fixed allocation, contextual bandit):
add a strategy name to `KNOWN_STRATEGIES`, a branch in `assignment._draw`, and
optional fields here — the orchestrator does not change.

## 7. Safety / guardrails

* Control events never reach this layer — the orchestrator skips them *before*
  calling the policy, so no action is assigned and no Intervention is created.
* The assigner never widens the eligible set. Exploration is strictly within
  `decision.candidates` (already filtered by the policy for already-tried
  actions, premium gating, max attempts), optionally narrowed further by
  `allowed_actions`. If `allowed_actions` excludes every policy candidate the
  assigner falls back to the policy's top-ranked action with `propensity = 1.0`
  and records a note — it never abandons an otherwise-workable event.
* `ε = 0` / disabled ⇒ byte-for-byte the previous orchestrator behaviour.

## 8. Determinism

Pass an explicit `random.Random` to `ActionAssigner(config, rng=…)` for tests,
or set `config.seed` so the per-decision RNG is `Random(f"{seed}:{event_id}")`
— reproducible and independent of event ordering within a batch.

## 9. What this does **not** give you — assumptions still unmet

epsilon-greedy / randomized assignment is *necessary* for action-level causal
estimation but far from *sufficient*. Still required before any uplift model:

1. **Valid, stable eligibility rules.** The eligible set must be a faithful,
   unchanging definition of "actions that could have been taken". If eligibility
   logic drifts between decisions, the logged propensity is for the wrong
   support set.
2. **Sufficient overlap / positivity.** Every action must have a non-trivial
   probability across the contexts where any action is used. Near-deterministic
   assignment (tiny min-propensity) makes IPW weights explode.
3. **Trustworthy outcome timestamps.** The current schema reconstructs the
   payment schedule and `as_of` ordering (successes carry no timestamp). Causal
   estimates need real event-time outcomes.
4. **Enough observations per action** — thousands of positives per action, not
   the current tens.
5. **No major hidden confounding outside the logged decision context.** IPW only
   adjusts for what was in the context at decision time. Anything that
   influenced both the (stochastic tie-break of the) assignment and the outcome,
   and was not logged, still biases the estimate.
6. **A control-arm response model** `P(Y | no intervention)` for true
   *intervention-vs-nothing* uplift, and **logged assignment propensities for a
   proper policy** (not just epsilon-greedy tie-breaking) for *action-vs-action*
   uplift with doubly-robust estimators.

Until those hold, treat `assignment-coverage` output as a *readiness check*, not
as evidence of a causal effect.
