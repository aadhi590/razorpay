# Stage report — Uplift / Causal Recovery Intelligence

> **SYNTHETIC BENCHMARK.** All figures come from
> `app/scripts/generate_data.py --randomized-assignment` (6 000 customers,
> seed 42), whose outcome process is a known closed form. Numbers below are a
> benchmark against a simulator, **not** real-world causal evidence.

Moves the system from `P(recovery | context, action)` (predictive, kept intact)
to `E[Y(a) − Y(0) | X]` (incremental recovery *caused* by an intervention).
Additive: the predictive ML layer, the experimentation layer, the rules policy
and all prior APIs are unchanged.

---

## 1. Files created / modified

### Created — `app/ml/uplift/`
| file | purpose |
|---|---|
| `config.py` | versions, arm list, cost table, split/threshold constants, `CONTROL_DESIGN_PROPENSITY` |
| `__init__.py` | package overview |
| `features/__init__.py` | **re-export** of the predictive feature contract + `BASELINE_FEATURES` (= `ALL_FEATURES` − `action`) |
| `dataset/decision_points.py` | control + attempt‑1 `decision_points` CTEs feeding the **shared** feature body; label & propensity SQL |
| `dataset/builder.py` | `build_uplift_dataset()` → `UpliftDataset` (one row per first decision) |
| `validation/propensity.py` | propensity range, overlap/positivity, ESS, stabilized IPW weights, per‑arm coverage |
| `validation/integrity.py` | reuses predictive DQ validator + arm domain / dedupe / split‑overlap / post‑decision‑column checks |
| `estimators/preprocessing.py` | `build_baseline_preprocessor()` (+ re‑export of the S‑learner preprocessor) |
| `estimators/base.py` | `BaseUpliftEstimator` interface, `UpliftPrediction` |
| `estimators/learners.py` | `SLearner`, `TLearner`, `build_estimator()` |
| `evaluation/metrics.py` | Qini curve + coefficient + bootstrap CI, uplift@k, observed lift by action, policy value (SNIPW), uplift calibration |
| `models/artifact.py` | `UpliftArtifact` (+ `save`/`load`, schema‑versioned, temp‑dir‑safe) |
| `registry/__init__.py` | cached `get_default_uplift_artifact()` / `try_load` / reset |
| `inference/predictor.py` | `UpliftModel` — baseline + per‑action uplift + `net_incremental_value_paise` |
| `training/train.py` | `run_uplift_training()` — dataset → validate → split → compare S/T learners → champion on held‑out policy value → test → artifact + report |
| `README.md` | full math, assumptions, worked example, architecture, limitations |
| `app/scripts/train_uplift_model.py` | CLI (`--fast` for CI) |
| `app/services/uplift_recovery_policy.py` | `UpliftRecoveryPolicy` (RecoveryPolicy protocol; uplift → ML → rules fallback) |
| `app/routes/uplift.py`, `app/schemas/uplift.py` | read‑only `GET /api/v1/uplift/model` + `/uplift-scores` |
| `tests/uplift/` (8 files), `tests/ml/conftest.py` | new tests + shared session fixtures |
| `docs/uplift_stage_report.md` | this document |

### Modified (additive, backward‑compatible)
| file | change |
|---|---|
| `app/ml/features/point_in_time.py` | + `compose_feature_sql(cte)` — attach a caller CTE to the **one** shared feature body (no fork) |
| `app/ml/models/artifact.py` | `save_artifact(..., update_latest=None)` — only refresh `*_latest.joblib` when writing into the real artifact dir (stops tests clobbering the committed artifact) |
| `app/services/policy_factory.py` | `POLICY_NAMES += ("uplift",)`; `resolve_policy` handles `uplift`/`causal` |
| `app/routes/__init__.py`, `app/main.py` | register `uplift_router` |
| `app/ml/README.md` | §11 rewritten to point at the built uplift layer |
| `tests/ml/test_features.py`, `test_dataset_split_validation.py`, `test_training_artifact.py` | consume session‑scoped fixtures (test‑perf only, no logic change) |

### Data
`generate_data.py --reset --customers 6000 --seed 42 --randomized-assignment`
(generator code unchanged; this is Phase 7 scale‑up).

---

## 2. Architecture (text)

```
Razorpay event
   │
   ▼ point-in-time context      app/ml/features/point_in_time.py  (ONE _FEATURE_BODY_SQL)
   ├──────────────────────────────────►  predictive  P(Y | X, a)      app/ml/  (unchanged)
   │
   ▼ compose_feature_sql(uplift decision_points CTE)
uplift dataset  — 1 row / first recovery decision
     control rows (arm='none', as_of = event.created_at)
     + attempt-1 treatment rows (arm = randomized action, same as_of)
     + arm, treatment, propensity (logged), recovered (label)
   │
   ├─ validation/propensity.py   range · overlap · positivity · ESS · IPW weights
   ├─ validation/integrity.py    predictive DQ validator + split-event separation + no post-decision cols
   ▼
grouped(recovery_event) + chronological(as_of) split   60 / 20 / 20
   │
   ▼ estimators/
   control response  mu_0(X)  = P(Y | X, no intervention)   — genuine control rows only
   T-learner: one calibrated classifier per arm  (BASELINE_FEATURES)
   S-learner: one calibrated classifier, action as indicator (ALL_FEATURES)
        uplift_a(X) = mu_a(X) - mu_0(X)
   │
   ▼ evaluation/metrics.py   Qini + bootstrap CI · uplift@k · observed lift/action · policy value (SNIPW) · uplift calibration
   │
   ▼ training/train.py   champion = max held-out policy-value gain vs random action
UpliftArtifact  →  artifacts/ml/uplift_recovery_{uplift_v1,latest}.joblib  + report.{json,md}
   │
   ▼ inference/predictor.py   UpliftModel.predict_for_event → baseline, per-action {mu_a, uplift, net_incremental_value}, recommended
   │
   ├──►  GET /api/v1/uplift/model  ·  GET /api/v1/uplift/recovery-events/{id}/uplift-scores   (read-only)
   │
   ▼ UpliftRecoveryPolicy  (?policy=uplift)   rank by net_incremental_value; decline if none > 0
        └── fallback ──►  MLRecoveryPolicy ──►  RulesBasedRecoveryPolicy
   ▼
RecoveryOrchestratorService   (unchanged; still enforces control / max-attempts / eligibility)
```

---

## 3. Exact mathematical definitions

Potential outcomes `Y(a)` (recovery under action `a`), `Y(0)` (recovery under no
intervention).

```
mu_0(X)      = P(Y = 1 | X, arm = none)                     control response model
mu_a(X)      = P(Y = 1 | X, arm = a)                        action response model
uplift_a(X)  = mu_a(X) − mu_0(X)                            ≈ E[Y(a) − Y(0) | X]   (CATE)
uplift_a(X) − uplift_b(X) = mu_a(X) − mu_b(X)               action-vs-action contrast

incremental_expected_revenue_paise = uplift_a(X) · payment_amount_paise
net_incremental_value_paise        = incremental_expected_revenue_paise − action_cost_paise
π_uplift(X) = argmax_a net_incremental_value_paise(X, a)     (declines if max ≤ 0)
```

Propensity (joint, per row):

```
control row      :  e = P(is_control) = 0.20                (design constant)
action-a row     :  e = P(treated) · P(a | eligible) = 0.80 · 0.25 = 0.20   (logged)
stabilized IPW w :  w_i = P(arm_i) / e_i                    (≈ 1 here)
```

Evaluation:

```
Qini(φ) = Y_t(φ) − Y_c(φ) · N_t(φ) / N_c(φ)                 top-φ population by predicted best-uplift
Qini coefficient = area(Qini − random line) / (|Qini(1)| / 2)
uplift@k          = mean(Y | treated, top-k) − mean(Y | control, top-k)
V_SNIPW(π)        = Σ_i 1{a_i = π(x_i)}/e_i · y_i  /  Σ_i 1{a_i = π(x_i)}/e_i     (held-out treated)
```

**Not claimed:** individual treatment effects; that any of this transfers to
production; that the S‑ vs T‑learner difference is statistically real.

---

## 4. Dataset statistics

| | value |
|---|---|
| dataset_version | `uds_28d868b23799` · feature_version `feat_v1` |
| rows / events / customers | 10 771 / 10 771 / 4 328 |
| decision time | `recovery_event.created_at` (both arms) |
| as_of range | 2025‑12‑06 → 2026‑09‑02 |
| split (event‑grouped, chronological) | train 6 463 (629 pos) · val 2 154 (229) · test 2 154 (203) |
| event overlap train/val/test | 0 / 0 / 0 |
| null features | only `cust_days_since_last_failure` (nullable, imputed) |

## 5. Control / treatment / action counts

| arm | rows | positives | recovery rate | mean propensity |
|---|---|---|---|---|
| **none (control)** | 2 214 | **58** | 0.0262 | 0.20 |
| retry | 2 181 | 175 | 0.0802 | 0.20 |
| sms_nudge | 2 134 | 241 | 0.1129 | 0.20 |
| whatsapp_nudge | 2 076 | 257 | 0.1238 | 0.20 |
| method_switch_prompt | 2 166 | 330 | 0.1524 | 0.20 |

Naive treatment−control lift: **+9.1 pp**. Monotone in the generator's action
effectiveness ordering.

## 6. Propensity & overlap diagnostics

| | value |
|---|---|
| propensity range | [0.20, 0.20] — 0 rows ≤ 0, 0 rows > 1, 0 below 1e‑3 |
| realized control fraction | 0.2056 (design 0.20, Δ 0.006) |
| stabilized IPW weight | mean 1.0005, max 1.028 |
| effective sample size | 10 766 / 10 771 = **99.95 %** |
| extreme weights (> 50) | 0 |
| per‑action distinct events | ≥ 2 076 each (all ≥ threshold) |

**Recommendation (emitted by the tool):** assignment is uniform‑randomized and
overlap is near‑perfect → IPW is **not required** for unbiased treatment/control
or action contrasts here; it is implemented and reported as a
production‑readiness diagnostic and a variance check on policy value.

## 7. Models compared

| learner | base | held‑out policy‑value gain vs random |
|---|---|---|
| **t_learner** | HistGradientBoosting | **+0.05500** ← champion |
| s_learner | HistGradientBoosting | +0.04979 |
| s_learner | LogisticRegression | +0.04697 |

All sigmoid‑calibrated (`CalibratedClassifierCV`, cv=3 — control arm is small).
X‑learner / DR‑learner deliberately **not** built: with ~58 control positives the
propensity + cross‑fitted outcome models they need would add variance, not remove
it. Interfaces are in place to add one when data supports it.

## 8. Evaluation metrics (held‑out test, n = 2 154; control positives = 15)

**Qini coefficient 0.337, bootstrap 95 % CI [0.193, 0.545].**
Total incremental responders (Qini @100 %): 128.

uplift@k (targeting by predicted best‑uplift; treated arm is *random action*):

| top‑k | observed uplift | mean predicted |
|---|---|---|
| 10 % | **+15.1 pp** | 0.209 |
| 20 % | +13.0 pp | 0.195 |
| 30 % | +12.7 pp | 0.185 |
| 50 % | +11.3 pp | 0.170 |

Observed lift by action (held‑out, model‑free):

| action | recovery rate | uplift vs control | z |
|---|---|---|---|
| control | 0.0347 | – | – |
| retry | 0.0824 | +0.0477 | 3.0 |
| sms_nudge | 0.1070 | +0.0722 | 4.1 |
| whatsapp_nudge | 0.1126 | +0.0779 | 4.4 |
| method_switch_prompt | 0.1339 | +0.0992 | 5.4 |

Uplift calibration: ranking is monotone (predicted‑uplift bins line up with
observed), but the T‑learner **over‑predicts magnitude** (predicted 0.09–0.20 vs
observed 0.02–0.13) — its small control model under‑estimates `mu_0`. Good enough
for *ordering* actions/customers; not for quoting an absolute pp number.

## 9. Uplift / policy‑value results

| policy (held‑out treated, SNIPW) | value |
|---|---|
| treat none (control rate) | 0.0347 |
| random action | 0.1092 |
| best single action (`method_switch_prompt` for everyone) | 0.1339 |
| **uplift policy π_uplift** | **0.1253** (gain vs random **+1.6 pp**, vs treat‑none **+9.1 pp**) |

π_uplift recommends `method_switch_prompt` for ~82 % of events, `whatsapp` ~14 %,
`sms` ~4 %. On this synthetic data one action dominates for almost everyone, so a
non‑personalized "always method_switch" is hard to beat on raw rate — the uplift
policy's value is (a) the *who to treat at all* decision and (b) cost‑awareness
(net value, not rate). Personalization headroom will only show with a generator
whose best action varies by context.

## 10. Champion & why

**`t_learner:hist_gb`** — highest held‑out **policy‑value gain over random action
assignment** (+0.055 vs +0.050 / +0.047), the metric that matches the business
objective (Phase 12). Not ROC‑AUC. **Caveat:** with 15 control positives in the
held‑out split, the S‑ vs T‑learner gap is inside the noise — the choice is
**not statistically decisive** and is recorded as such in the artifact.

## 11. Example per‑customer inference

`GET /api/v1/uplift/recovery-events/{id}/uplift-scores` →

```json
{
  "recovery_event_id": 12398,
  "available": true,
  "model_version": "uplift_v1",
  "baseline_probability": 0.041,
  "amount_paise": 199900,
  "actions": [
    {"action": "method_switch_prompt", "treatment_probability": 0.198, "uplift": 0.157,
     "cost_paise": 150, "incremental_expected_revenue_paise": 31384.3,
     "net_incremental_value_paise": 31234.3, "rank": 1},
    {"action": "whatsapp_nudge", "treatment_probability": 0.141, "uplift": 0.100,
     "cost_paise": 80, "net_incremental_value_paise": 19910.0, "rank": 2},
    {"action": "sms_nudge",  "...": "...", "rank": 3},
    {"action": "retry",      "...": "...", "rank": 4}
  ],
  "recommended_action": "method_switch_prompt"
}
```

Worked contrast (README §8): baseline 0.20 + WhatsApp 0.55 → uplift 0.35 on
₹1 000 = **₹349 net**; vs baseline 0.80 + SMS 0.82 → uplift 0.02 = **₹19.8 net**.
The uplift policy ranks the first ~17× higher even though the second has the
higher raw recovery probability.

## 12. API endpoints added

| method | path | returns |
|---|---|---|
| GET | `/api/v1/uplift/model` | artifact identity, champion reason, dataset stats, propensity diagnostics, test evaluation, limitations |
| GET | `/api/v1/uplift/recovery-events/{id}/uplift-scores` | baseline probability, per‑action `{treatment_probability, uplift, incremental_expected_revenue, net_incremental_value, rank}`, `recommended_action` |

Orchestrator: `?policy=uplift` on `POST .../orchestrate` and
`POST /api/v1/orchestrator/run` (default stays `rules`).

## 13. Test results

`tests/uplift/` (8 files, ~50 cases) + `tests/ml/` — **90 passed, 0 failed.**
Full `tests/` suite green (see run `byp8gmz44`). Coverage: dataset construction,
point‑in‑time leakage, split‑event separation, control model, S/T‑learner
training, uplift calculation, propensity validation, overlap diagnostics, unseen
categories / missing values, artifact save/load, inference, action ranking,
control behaviour, fallback chain, rules policy unchanged, ML policy unchanged,
orchestrator regression, API.

Test‑suite performance (side‑fix requested mid‑stage): `tests/ml/` +
`tests/uplift/` went from **2 965 s → 581 s** by session‑scoping the expensive
point‑in‑time build (test‑layer only; no model/eval logic touched).

## 14. Migration changes

**None.** No schema change, no Alembic migration. Propensity is read from the
existing `AgentEvent.input_context` JSON (`assignment.propensity`), written by the
generator's `--randomized-assignment` mode. Artifacts are filesystem‑only
(`uplift_recovery_uplift_v1` / `_latest`, `feature_version=feat_v1`,
`dataset_version=uds_<hash>`).

## 15. Known limitations

1. **Synthetic data** — benchmark only, not causal evidence.
2. **Small control arm** — 2 214 rows / 58 positives (15 in the held‑out split).
   `mu_0` and everything built on it are high‑variance; the champion selection is
   noisy; the T‑learner over‑predicts uplift magnitude.
3. **Random action in the treated arm** — Qini / uplift@k measure the value of
   choosing *who* to treat while treating with a *random* action (a lower bound
   on choosing *what*). Policy value conditions on action match.
4. **First decision only** — escalation attempts 2–3 excluded; nothing said about
   sequential strategy.
5. **Reconstructed timestamps** — inherited from the predictive layer (payment
   schedule, ~uniform failure‑time sampling).
6. **No hidden‑confounding adjustment** beyond the logged context — valid here
   *only* because assignment is known‑randomized; the load‑bearing untestable
   assumption in any real deployment.
7. **Little personalization headroom** on this generator (one action dominates
   for nearly everyone).
8. **Feature build is slow** (~2–3 min) — offline only.

## 16. Exact next step toward the autonomous AI agent

Build the LLM decision agent as a new component that, per open recovery event,
calls **both** `GET /api/v1/ml/.../action-scores` (predictive) and
`GET /api/v1/uplift/.../uplift-scores` (causal), plus
`GET /api/v1/analytics/assignment-coverage` (data‑readiness), and reasons over:
`baseline_probability`, per‑action `uplift` and `net_incremental_value_paise`,
the propensity/overlap diagnostics, and the documented assumptions — then either
accepts `recommended_action`, overrides it with a written rationale, or declines
to intervene. It writes its decision, the rationale, and the model versions it
consulted to `AgentEvent.input_context`, and the existing orchestrator +
experimentation layer execute and log exactly as today. The agent consumes the
incremental‑value frame; it does not replace any model.

Do **not** in that stage: build Razorpay API integration, remove the predictive
or experimentation layers, or let the agent write outside `AgentEvent`.
