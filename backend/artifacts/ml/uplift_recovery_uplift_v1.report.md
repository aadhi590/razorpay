# uplift_recovery uplift_v1 -- uplift training report

_Created: 2026-09-02T09:34:57.807328+00:00_

> **SYNTHETIC BENCHMARK.** Trained on synthetic data from app/scripts/generate_data.py --randomized-assignment, whose outcome process is a known closed form. Every uplift / Qini / policy-value number is a SYNTHETIC BENCHMARK, not real-world causal evidence. The control arm is small (tens of positives), so treat bootstrap intervals, not point estimates, as the result.

## Dataset (one row per first recovery decision)
- rows: 10771  |  events: 10771  |  customers: 4328
- control: 2214 rows (58 recovered, rate 0.026197)
- treatment: 8557 rows, rate 0.117214  (naive lift 0.091017)

### Per-arm
| arm | rows | positives | recovery rate | mean propensity |
|---|---|---|---|---|
| none | 2214 | 58 | 0.026197 | 0.2 |
| retry | 2181 | 175 | 0.080238 | 0.2 |
| sms_nudge | 2134 | 241 | 0.112933 | 0.2 |
| whatsapp_nudge | 2076 | 257 | 0.123796 | 0.2 |
| method_switch_prompt | 2166 | 330 | 0.152355 | 0.2 |

## Propensity / overlap
- realized control fraction: 0.205552 (design 0.2)
- stabilized-IPW effective sample size: 10766.0 / 10771 (0.9995)
- recommendation: Assignment is uniform-randomized and overlap is near-perfect (ESS = 100% of n). Inverse-propensity weighting is NOT required for unbiased treatment/control and action contrasts on this dataset; it is implemented and reported only as a production-readiness diagnostic and as a variance check on the policy-value estimates.

## Learner comparison (validation policy-value gain vs random action)
| learner | selection score |
|---|---|
| t_learner:hist_gb | +0.05500 |
| s_learner:hist_gb | +0.04979 |
| s_learner:logreg | +0.04697 |

**Champion: t_learner:hist_gb** -- selected 't_learner:hist_gb': highest held-out policy-value gain over random action assignment (+0.05500 recovery-rate points); ties broken toward the Qini coefficient. Scores: t_learner:hist_gb=+0.05500, s_learner:hist_gb=+0.04979, s_learner:logreg=+0.04697

## Held-out test evaluation
- n=2154 (treated 1722, control 432; control positives 15)
- Qini coefficient: 0.3373  (bootstrap 95% CI [0.1932, 0.5451])
- total incremental responders (Qini @100%): 128.208

### Policy value (self-normalized IPW on held-out treated)
- treat-none (control rate): 0.034722
- random action: 0.109175
- best single action (method_switch_prompt): 0.133929
- **uplift policy: 0.12528**  (gain vs random 0.016104, vs treat-none 0.090557)
- recommended-action mix: {'method_switch_prompt': 1404, 'whatsapp_nudge': 248, 'sms_nudge': 70}

### Observed lift by action (held-out)
| action | n | recovery rate | uplift vs control | z |
|---|---|---|---|---|
| control | 432 | 0.034722 | - | - |
| retry | 437 | 0.08238 | 0.047658 | 3.010719 |
| sms_nudge | 402 | 0.106965 | 0.072243 | 4.069101 |
| whatsapp_nudge | 435 | 0.112644 | 0.077921 | 4.44456 |
| method_switch_prompt | 448 | 0.133929 | 0.099206 | 5.40817 |

- reliability: PASS: enough held-out control outcomes for a directional read.

## Known limitations
- SYNTHETIC DATA. Results are a benchmark against a known generator, not evidence of a real causal effect.
- Control arm is small: 2214 rows / 58 natural recoveries. mu_0 and every uplift number built on it are high-variance.
- Treatment rows carry a UNIFORMLY RANDOM first action, not the policy's choice, so Qini / uplift@k estimate the value of targeting WHO to treat with a random action -- a lower bound on the value of also choosing WHAT.
- First-decision only: escalation attempts 2-3 are excluded, so the model says nothing about sequential recovery strategy.
- Decision time is the recovery event's creation time for both arms; the generator's reconstructed payment schedule and ~uniform failure-time sampling still apply (see app/ml/README.md).
- No hidden-confounding adjustment beyond the logged decision context; valid here only because assignment is known-randomized.
