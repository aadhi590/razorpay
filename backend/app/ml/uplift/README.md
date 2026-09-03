# Uplift / Causal Recovery Intelligence (`app/ml/uplift/`)

> **SYNTHETIC BENCHMARK.** Every number this layer produces comes from
> `app/scripts/generate_data.py --randomized-assignment`, whose outcome process
> is a known closed form. These are benchmarks against a simulator, **not**
> real-world causal evidence. The control arm is small (tens of natural
> recoveries), so the S-/T-learner choice is not statistically decisive and the
> Qini point estimate is inside its own bootstrap interval. See §10.

## 1. Why predictive ML is not enough

The predictive model (`app/ml/`) estimates

```
P(Y = 1 | X, action)          "how likely is recovery if we take this action"
```

Ranking actions by that (or by `P·amount − cost`) over-spends on customers who
would have recovered **anyway**. A payment with `P(recover | SMS) = 0.82` looks
great — but if `P(recover | nothing) = 0.80`, the SMS bought 0.02 of recovery
probability. The money should go where the intervention *causes* the recovery.

This layer estimates the causal quantity:

```
uplift_a(X) = E[Y(a) − Y(0) | X] = mu_a(X) − mu_0(X)
```

`Y(a)` and `Y(0)` are potential outcomes (recovery under action `a` vs under no
intervention). We estimate the **conditional average** effect, never an
individual one — see §7.

## 2. Why control is necessary

`mu_0(X) = P(Y = 1 | X, no intervention)` cannot be read off treated customers:
by construction they *were* intervened on. It needs genuine **control**
observations — recovery events the orchestrator deliberately left alone. The
generator holds out `CONTROL_GROUP_FRACTION = 0.20` of failed payments as
control (no `Intervention` rows; the payment either recovers naturally or does
not). We **never** treat a failed treatment attempt as pseudo-control and we
**never** fabricate a counterfactual label.

## 3. Why randomized assignment helps

Among treated events, which action was taken was historically a deterministic
function of context — so "what would WhatsApp have done here?" is unanswerable
(action is perfectly confounded with context). `generate_data.py
--randomized-assignment` draws the first action **uniformly at random** over the
eligible set and logs the exact probability (`propensity`) on the `AgentEvent`.
With random assignment, the action groups are exchangeable given context, so
`mu_a` is identified from the group that got `a`.

## 4. What propensity means

`propensity` = P(this arm/action cell was assigned | context, mechanism),
evaluated *after* eligibility filtering. In this dataset:

| cell | propensity |
|---|---|
| control | `P(is_control) = 0.20` (design constant, cross-checked against the realized fraction) |
| first action `a` | `P(treated) · P(a | eligible) = 0.80 · 0.25 = 0.20` |

Because assignment is **known uniform-randomized**, inverse-propensity weighting
(IPW) is **not required** for unbiased treatment/control or action contrasts on
this dataset — the effective sample size under stabilized IPW is ~100 % of `n`.
The propensity code (`validation/propensity.py`) is still implemented and run: it
(a) verifies the "known randomized" claim rather than assuming it, and (b) is the
diagnostic a real, non-randomized deployment would depend on. The policy-value
estimates *do* use self-normalized IPW as a variance check.

## 5. The dataset (`dataset/`)

One row per **first recovery decision**, `as_of = recovery_event.created_at`
(identical decision time for both arms → point-in-time features are comparable
and no post-decision info leaks):

```
control arm    -> every is_control event,      arm = 'none'
treatment arm  -> the attempt-1 intervention,  arm = <randomized action>
```

"Attempt 1" is by `interventions.id` (insertion order), **not** `executed_at` —
the generator's per-attempt latency grows with the attempt number, so a later
attempt can carry an earlier `executed_at`. Escalation attempts 2–3 are a
separate decision and are excluded from v1.

Features are the predictive layer's contract **verbatim**
(`app/ml/uplift/features` re-exports it; `compose_feature_sql` attaches a new
`decision_points` CTE to the *same* feature body). `BASELINE_FEATURES` =
`ALL_FEATURES` minus `action` (constant within the control model and each
T-learner slice).

Columns added: `arm`, `treatment`, `raw_action_propensity` (logged),
`propensity` (joint), `recovered` (label), `assignment_strategy`.

## 6. The models (`estimators/`)

Both are sklearn `Pipeline`s (own their preprocessing), sigmoid-calibrated
(`CalibratedClassifierCV`), `random_state = 42`, no new dependencies.

**Control response model** — `mu_0`: a calibrated classifier over
`BASELINE_FEATURES`, trained on **control rows only**. First-class: it is the
`arm = 'none'` sub-model of the T-learner and the `action = 'none'` evaluation of
the S-learner, and it is versioned inside the same artifact.

**T-learner** — one calibrated classifier per arm (`none`, `retry`,
`sms_nudge`, `whatsapp_nudge`, `method_switch_prompt`) over `BASELINE_FEATURES`.
`uplift_a(X) = mu_a(X) − mu_0(X)`. Each arm gets its own response surface;
the tiny control arm gets its own small model (higher variance).

**S-learner** — one calibrated classifier over `ALL_FEATURES` with `action`
(incl. `'none'`) as the treatment indicator. `mu_a(X)` = score with
`action = a`; `mu_0(X)` = score with `action = 'none'`. Pools all rows (lower
variance) but regularization can shrink a small uplift toward zero.

**X-learner / DR-learner:** not implemented in v1. With ~2.6 % control base
rate and ~58 control positives the propensity model and the cross-fitted
outcome models an X-/DR-learner needs would be noisier than the T-learner, not
less — added complexity for no defensible gain on this data. The
interfaces (`BaseUpliftEstimator`, logged propensity) are in place to add one
when the data supports it.

### Action-vs-action

`uplift_a − uplift_b = mu_a − mu_b` is defensible here (both arms randomized,
same support), and the API returns every action's uplift so a caller can
compare. The champion policy already ranks actions against each other by net
incremental value.

## 7. Assumptions still required (honest list)

Identification of `uplift_a(X)` as a causal effect needs, and this setup gives:

1. **Randomized assignment** — ✅ by generator design, propensity logged.
2. **Positivity / overlap** — ✅ every cell has propensity 0.20; ESS ≈ n.
3. **No unmeasured confounding of assignment and outcome** — ✅ *only because*
   assignment is randomized; in production this is the load-bearing, untestable
   assumption.
4. **SUTVA / no interference** — assumed (one customer's treatment doesn't move
   another's outcome).
5. **Consistency** — the logged `action` is the treatment actually delivered.
6. **Correct, stable eligibility** — the eligible set is a faithful definition of
   "actions that could have been taken".

What we **cannot** claim: individual treatment effects; that the S-/T-learner
difference is real; that any of this transfers to production data;
sequential-strategy effects (v1 is first-decision only).

## 8. How uplift becomes incremental revenue (`inference/`, Phase 12)

```
baseline          = mu_0(X)
treated           = mu_a(X)
uplift_a          = treated − baseline
incremental_expected_revenue = uplift_a · payment_amount_paise
net_incremental_value        = incremental_expected_revenue − action_cost_paise
```

`UpliftRecoveryPolicy` ranks eligible actions by `net_incremental_value` and, by
default, **declines to intervene** when no action clears
`min_net_incremental_value_paise = 0` — the economically-honest outcome.

### Worked example

| | baseline `mu_0` | action `mu_a` | uplift | payment | incr. revenue | cost | **net** |
|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.55 (WhatsApp) | **0.35** | ₹1,000 | ₹350.00 | ₹0.80 | **₹349.20** |
| B | 0.80 | 0.82 (SMS) | **0.02** | ₹1,000 | ₹20.00 | ₹0.20 | **₹19.80** |

Customer B has the higher raw recovery probability under the action (0.82 vs
0.55) — a predictive policy might rank B first. The uplift policy ranks A ~17×
higher because A's recovery is *caused* by the intervention while B would
mostly have recovered anyway.

## 9. Architecture

```
Razorpay event
   │
   ▼ point-in-time context   (app/ml/features/point_in_time.py — ONE feature body)
   ├─────────────────────────────► predictive P(Y|X,a)   (app/ml/  — unchanged)
   │
   ▼ uplift decision points   (app/ml/uplift/dataset — control + attempt-1 rows)
uplift dataset (1 row / decision, arm, propensity, label)
   │  validation/propensity.py   (overlap, positivity, ESS, IPW weights)
   │  validation/integrity.py    (reuses predictive DQ validator + split integrity)
   ▼
grouped(recovery_event) + chronological split
   │  estimators/  (control model  +  {S-learner, T-learner} × {hist_gb, logreg})
   ▼
learner comparison on validation  (evaluation/metrics.py: Qini, uplift@k, policy value)
   │  training/train.py  — champion = best held-out policy-value gain vs random
   ▼
UpliftArtifact  (models/artifact.py -> artifacts/ml/uplift_recovery_*.joblib)
   │  inference/predictor.py  (UpliftModel: baseline + per-action uplift + economics)
   ▼
UpliftRecoveryPolicy  (app/services/uplift_recovery_policy.py)
   │      └── fallback ──► MLRecoveryPolicy ──► RulesBasedRecoveryPolicy
   ▼
RecoveryOrchestratorService  (unchanged)  +  GET /api/v1/uplift/...
```

## 10. Limitations

- **Synthetic data.** Benchmark only.
- **Small control arm.** ~2,200 control rows / ~58 natural recoveries total
  (~12 in the held-out split). `mu_0` and every uplift built on it are
  high-variance; the champion is selected on a noisy validation signal.
- **Random action in the treated arm.** Qini / uplift@k measure the value of
  targeting *who* to treat while treating them with a *random* action — a lower
  bound on the value of also choosing *what*. `policy_value` conditions on action
  match and is the number that reflects "who and what".
- **First decision only.** Nothing is said about escalation strategy.
- **Reconstructed timestamps** for the payment schedule (inherited from the
  predictive layer).
- **Feature build is slow** (~1–3 min: many point-in-time correlated
  subqueries). Offline-only; the CLI caches nothing.

## 11. Usage

```bash
# data (larger scale + randomized assignment)
python -m app.scripts.generate_data --reset --customers 6000 --seed 42 --randomized-assignment

# train (never runs implicitly / at API startup)
python -m app.scripts.train_uplift_model          # full
python -m app.scripts.train_uplift_model --fast   # CI/tests
```

```python
from app.ml.uplift.inference.predictor import UpliftModel
m = UpliftModel.load()                       # raises UpliftModelUnavailable if absent
m.predict_for_event(conn, recovery_event_id) # -> EventUplift(baseline, [ActionUplift...], recommended)
```

API (read-only): `GET /api/v1/uplift/model`,
`GET /api/v1/uplift/recovery-events/{id}/uplift-scores`.
Orchestrator: `POST .../orchestrate?policy=uplift` and
`POST /api/v1/orchestrator/run?policy=uplift` (default stays `rules`; `uplift`
falls back to `ml` then `rules` if the artifact is missing/invalid — the
workflow never breaks).

## 12. Next step toward the autonomous agent

The agent will call `GET /api/v1/uplift/recovery-events/{id}/uplift-scores`
alongside the predictive `action-scores`, reason over `baseline_probability`,
per-action `uplift`, `net_incremental_value_paise` and the documented
assumptions, and either accept the `recommended_action`, override it with a
rationale, or decline — writing its decision (and which model versions it
consulted) to `AgentEvent.input_context`. The uplift layer gives the agent the
*incremental-value* frame; it does not make the decision.
