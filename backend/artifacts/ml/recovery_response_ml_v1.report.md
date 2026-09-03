# recovery_response ml_v1 -- training report

_Created: 2026-09-02T09:40:20.943235+00:00_

> **SYNTHETIC BENCHMARK.** Trained on synthetic data from a generator with a known outcome process. Metrics are a SYNTHETIC BENCHMARK, not real-world predictive performance; the model may be recovering the simulator's rules.

## Dataset
- rows: 12207  |  recovery events: 8557  |  customers: 3965
- positives: 1221 (0.1000)
- action counts: {'method_switch_prompt': 3100, 'retry': 3066, 'sms_nudge': 3032, 'whatsapp_nudge': 3009}
- as_of range: 2025-12-06 20:42:23.839240+00:00 .. 2026-09-06 18:55:03.757206+00:00

## Split (grouped by recovery event, chronological)
- train: 7304 rows / 5134 events / 729 pos (0.099808), 2025-12-06 20:42:23.839240+00:00 .. 2026-06-11 07:00:59.846900+00:00
- validation: 2428 rows / 1711 events / 267 pos (0.109967), 2026-06-08 15:44:36.020814+00:00 .. 2026-07-26 21:51:34.999858+00:00
- test: 2475 rows / 1712 events / 225 pos (0.090909), 2026-07-22 22:34:17.051441+00:00 .. 2026-09-06 18:55:03.757206+00:00
- event overlap: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- customer overlap: {'train_val': 813, 'train_test': 618, 'val_test': 546, 'note': 'customer overlap is expected and does not leak the label; history features are point-in-time per row'}

## Model comparison (validation)
| model | ROC-AUC | PR-AUC | log loss | Brier | ECE |
|---|---|---|---|---|---|
| logreg | 0.7007124943889551 | 0.22719118923820436 | 0.32171648665015695 | 0.09279309508626453 | 0.00869 |
| hist_gb | 0.6839946133968356 | 0.20278531440448277 | 0.3259550007397379 | 0.09395882628517417 | 0.011229 |
| random_forest | 0.68546605036162 | 0.1979659147861574 | 0.32691105729750936 | 0.0942252024741466 | 0.014 |

**Selected: logreg** -- best validation PR-AUC=0.2272; among models within 0.02 PR-AUC (['logreg']) selected 'logreg' for lowest validation ECE (0.0087)

## Test metrics (untouched)
- n=2475  positives=225 (0.090909)
- ROC-AUC=0.7011466666666666  PR-AUC=0.18616203781712584
- log loss=0.2843502283467402  Brier=0.07917235816684175
- precision=0.0  recall=0.0  F1=0.0  (threshold 0.5)
- confusion matrix: {'tn': 2250, 'fp': 0, 'fn': 225, 'tp': 0}

## Calibration (test)
- method: sigmoid  (only 729 positives in train (< 1000); isotonic would overfit)
- uncalibrated: Brier=0.2264209760984707  ECE=0.361027
- calibrated:   Brier=0.07917235816684175  ECE=0.01372

## Per-action (test)
- method_switch_prompt: n=644 pos=72 observed=0.111801 predicted=0.130082 
- retry: n=637 pos=41 observed=0.064364 predicted=0.074784 
- sms_nudge: n=580 pos=53 observed=0.091379 predicted=0.09802 
- whatsapp_nudge: n=614 pos=59 observed=0.096091 predicted=0.112935 

## Top permutation importances (test)
- failure_reason: 0.06034 ± 0.00480
- attempt_number: 0.02134 ± 0.01008
- cust_prior_successful_payments: 0.01997 ± 0.00778
- action: 0.01892 ± 0.00715
- already_tried_sms_nudge: 0.00711 ± 0.00375
- already_tried_retry: 0.00669 ± 0.00577
- hours_since_failure: 0.00650 ± 0.00861
- already_tried_method_switch_prompt: 0.00527 ± 0.00366
- already_tried_whatsapp_nudge: 0.00406 ± 0.00636
- cust_prior_failed_payments: 0.00273 ± 0.00338
