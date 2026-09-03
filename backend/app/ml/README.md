# Recovery-Response Model (`app/ml/`)

> **SYNTHETIC BENCHMARK.** Everything here is trained and evaluated on data from
> `app/scripts/generate_data.py`, whose outcome process is a known closed form.
> Metrics are a *synthetic benchmark*, **not** real-world predictive
> performance — the model is partly re-learning the simulator's rules. Do not
> tune the model to reproduce the generator.

## 1. Objective

For a specific failed payment and a specific recovery action, predict

```
P(recovery | point-in-time customer/payment context, action)
```

The action is part of the input, not a filter on the output — the model scores
every candidate action for the same context. It is a **quantitative decision
tool**, not the agent: `MLRecoveryPolicy` ranks actions by predicted expected
value and the orchestrator persists the choice; a future AI agent will sit on
top of this.

## 2. Architecture / pipeline

```
PostgreSQL
   │  app/ml/features/point_in_time.py   (ONE SQL body, train == inference)
   ▼
point-in-time feature frame
   │  app/ml/datasets/builder.py + coverage.py
   ▼
training dataset  (1 row per observed intervention: context + action + label)
   │  app/ml/training/validation.py      (data-quality / leakage — FAILS LOUDLY)
   ▼
grouped chronological split  (app/ml/training/splitting.py)
   │  app/ml/preprocessing/pipeline.py   (ColumnTransformer, packaged in the estimator)
   ▼
model comparison  (app/ml/models/registry.py: logreg / hist_gb / random_forest)
   │  app/ml/training/train.py
   ▼
probability calibration  (sigmoid; CalibratedClassifierCV)
   ▼
evaluation on untouched test  (app/ml/evaluation/metrics.py)
   ▼
model artifact  (app/ml/models/artifact.py -> artifacts/ml/*.joblib)
   │  app/ml/inference/predictor.py      (RecoveryModel)
   ▼
MLRecoveryPolicy  (app/services/ml_recovery_policy.py) ── rules fallback
   │
   ▼
RecoveryOrchestratorService  (unchanged)
```

`app/ml/monitoring/drift.py` provides a training-time feature baseline, a PSI
drift check, and an append-only prediction log (`monitoring readiness`).

## 3. Features (`feat_v1`)

Every feature is available **strictly before** the decision time `T`
(`intervention.executed_at` in training, `now()` at inference). Enforced in SQL:
every subquery filters `< T` / `executed_at < T` / `failed_at < T` /
`due_at < T`. The feature SQL never touches the `outcomes` table.

**Numeric (19):** `amount_paise`, `subscription_amount`, `priority`,
`hours_since_failure`, `subscription_tenure_days`, `attempt_number`,
`already_tried_{retry,sms_nudge,whatsapp_nudge,method_switch_prompt}`,
`cust_prior_total_payments`, `cust_prior_failed_payments`,
`cust_prior_successful_payments`, `cust_prior_recovered_payments`,
`cust_prior_success_ratio`, `cust_prior_failure_ratio`,
`cust_days_since_last_failure` (nullable → imputed + indicator),
`cust_failures_last_90d`.

**Categorical (3):** `failure_reason`, `action` (the treatment variable),
`experiment_intervention_type`.

**Label:** `recovered` = `outcomes.payment_recovered` (the only column read from
`outcomes`).

### Leakage prevention

- Forbidden sources (`outcomes`, `payment_recovered`, `recovered_amount_paise`,
  `recovery_time_seconds`, `observed_at`) are asserted absent from the feature
  SQL at validation time.
- `hours_since_failure ≥ 0`, `attempt_number ≥ 1`, non-negative counts, ratios
  in `[0,1]` — violations **fail** training.
- The train/val/test split is verified to have **zero** shared recovery events.
- Customer-history features exclude the current payment and use a reconstructed
  payment schedule (`subscription.started_at + rank*30d`) because successful
  payments carry no timestamp in the current schema.

### Dropped on purpose

`customer_tenure_days` — the generator's `customer.created_at` is not
constrained to precede the customer's activity (≈38% of rows would be
negative). A production system with a real onboarding timestamp should add it
back.

## 4. Dataset construction

One row per **observed** intervention: `(event context @ executed_at, action,
recovered)`. We **never** fabricate a counterfactual outcome for an action that
was not tried. Control events (no intervention) are excluded and reserved for
uplift. A recovery event can contribute multiple rows (escalation), and rows
from one event never cross the split boundary.

## 5. Split strategy

Grouped (by `recovery_event_id`) + chronological (by the event's earliest
`as_of`): oldest 60% of events → train, next 20% → validation, newest 20% →
test. Customers may appear in more than one split — expected, and not a leak
because every history feature is point-in-time per row. **Caveat:** `as_of`
derives from `payment.failed_at`, which the generator samples ~uniformly over a
year, so the ordering is the simulator's sampling order, not real business time.

## 6. Models & selection

`LogisticRegression` (interpretable baseline, always kept), sklearn
`HistGradientBoostingClassifier`, `RandomForestClassifier`. All are full
`Pipeline`s (own their preprocessing), `random_state=42`, `class_weight`
balanced. Selection rule: best **validation PR-AUC**; among models within `0.02`
PR-AUC of the best, pick the lowest **ECE** (calibration). No LightGBM/XGBoost —
`HistGradientBoosting` covers boosting without a new dependency.

## 7. Calibration

Probabilities drive decisions, so they are calibrated with **Platt / sigmoid**
scaling (`CalibratedClassifierCV`, internal CV). Isotonic is rejected while the
training positive count is `< 1000` (it would overfit). The report shows
uncalibrated vs calibrated Brier / ECE and a reliability table.

## 8. Artifacts (`artifacts/ml/`)

- `recovery_response_ml_v1.joblib` — the `ModelArtifact` (fitted calibrated
  estimator + full feature contract + dataset/split/DQ/metrics/calibration/
  interpretability/baseline/config/versions).
- `recovery_response_latest.joblib` — stable "latest" pointer (a copy).
- `recovery_response_ml_v1.report.{json,md}` — human/machine training report.

Versioning is filesystem metadata only (`model_version=ml_v1`,
`feature_version=feat_v1`, `dataset_version=ds_<hash>`), no DB migration.

## 9. Usage

```bash
# train (never runs implicitly / at API startup)
python -m app.scripts.train_recovery_model            # full
python -m app.scripts.train_recovery_model --fast     # CI/tests
```

```python
from app.ml.inference.predictor import RecoveryModel
m = RecoveryModel.load()                       # raises ModelUnavailable if absent
m.predict_recovery(features, "whatsapp_nudge") # -> float in [0,1]
m.predict_all_actions(features)                # -> {action: prob}
m.predict_for_event(conn, recovery_event_id)   # -> {action: ActionScore(prob, cost, EV)}
```

API (read-only): `GET /api/v1/ml/model`,
`GET /api/v1/ml/recovery-events/{id}/action-scores`.
Orchestrator: `POST /api/v1/recovery-events/{id}/orchestrate?policy=ml` and
`POST /api/v1/orchestrator/run?policy=ml` (default `rules`; `ml` falls back to
`rules` if the artifact is missing/invalid — the workflow never breaks).

## 10. Limitations

- **Synthetic data.** Metrics are a benchmark, not production performance.
- **Weak action signal.** Action assignment in the training data is
  policy/random driven, not randomized; combined with a small effectiveness
  spread on a ~10% base rate, the model's *action-level* discrimination is
  limited (see `interpretability` in the report). Action-level decisions need
  randomized assignment or logged propensities.
- **Reconstructed timestamps** for the payment schedule and for `as_of`
  ordering.
- Small positive count (~186) → per-action metrics for the thinner actions are
  noisy; calibration is sigmoid-only.

## 11. Uplift / causal layer — now built (`app/ml/uplift/`)

The uplift layer is implemented and coexists with this predictive model
(neither replaces the other). It reuses **this layer's exact feature contract**
(`app/ml/uplift/features` re-exports `schema.py`; `compose_feature_sql` attaches
a new `decision_points` CTE to the *same* `_FEATURE_BODY_SQL`) — there is no
second feature definition.

1. Treatment response `mu_a(X) = P(Y | X, action)` — this predictive model *and*
   the per-arm sub-models of the uplift T-/S-learner.
2. Control response `mu_0(X) = P(Y | X, no intervention)` — a first-class,
   versioned model trained on genuine control events only.
3. `uplift_a(X) = mu_a(X) − mu_0(X)`; `net_incremental_value =
   uplift_a · amount − cost`; `UpliftRecoveryPolicy` (`?policy=uplift`) ranks on
   that, falling back to `ml` → `rules`.

Data prerequisite is satisfied by `generate_data.py --randomized-assignment`
(uniform first-action assignment, propensity logged) at 6k-customer scale.
`GET /api/v1/analytics/assignment-coverage` remains the readiness check;
`GET /api/v1/uplift/...` serves the causal estimates.

See **`app/ml/uplift/README.md`** for the mathematical definitions, propensity /
overlap diagnostics, model comparison, Qini / policy-value results, the honest
list of remaining assumptions, and what cannot be claimed from synthetic data.
