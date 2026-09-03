# Razorpay Test Mode integration & end-to-end recovery flow

Stage status: **integration + full mocked/locally-signed verification complete,
and the one-time real Razorpay Test Mode end-to-end smoke run has been executed.**
A real Test Mode Payment Link was created, paid with a test card, and the
`payment_link.paid` webhook confirmed recovery end to end
(recovery event 18499 → intervention 19696 → Payment Link `plink_TXIOWJC2UJRgcc`
→ payment `pay_TXIcHbfpqUy35R` → Outcome, `recovered_amount_paise = 49900` from
the Razorpay payment entity, event closed). See [§10](#10-what-is-external-vs-simulated).

---

## 1. Architecture

```
                 ┌─────────────────────────────────────────────┐
   Gemini agent  │ app/agent/  (unchanged loop, non-deterministic)
                 └───────────────┬─────────────────────────────┘
                                 │  execute_recovery_action / observe_recovery_outcome
                 ┌───────────────▼─────────────────────────────┐
   guardrails    │ app/agent/guardrails.py  (authoritative)     │
                 └───────────────┬─────────────────────────────┘
                 ┌───────────────▼─────────────────────────────┐
   exec service  │ app/services/recovery_execution.py           │
                 │  • build request from AUTHORITATIVE payment  │
                 │  • idempotent link create / reuse           │
                 │  • persist correlation ids on Intervention  │
                 └───────────────┬─────────────────────────────┘
                 ┌───────────────▼─────────────────────────────┐
   adapter       │ app/integrations/razorpay/  (HTTP only)      │
                 └───────────────┬─────────────────────────────┘
                                 ▼
                       Razorpay Test Mode API

   Razorpay ──webhook──▶ POST /api/v1/webhooks/razorpay
                          → app/services/razorpay_webhook.py
                            • verify signature (HMAC) BEFORE parse
                            • correlate → Intervention / RecoveryEvent
                            • create the single Outcome, close event
                            • idempotent (processed_webhook_events PK)
```

The policy layer (`RulesBasedRecoveryPolicy`, `MLRecoveryPolicy`,
`UpliftRecoveryPolicy`, `ActionAssigner`), the orchestrator, analytics, and every
pre-existing route are untouched. Razorpay is an **additional execution
capability**, selected only when the agent runs `dry_run=false` with valid test
config.

## 2. Test Mode configuration

| Env var | Default | Notes |
|---|---|---|
| `RAZORPAY_KEY_ID` | – | must start with `rzp_test_` |
| `RAZORPAY_KEY_SECRET` | – | never logged |
| `RAZORPAY_WEBHOOK_SECRET` | – | verifies `X-Razorpay-Signature` |
| `RAZORPAY_BASE_URL` | `https://api.razorpay.com/v1` | HTTPS enforced |
| `RAZORPAY_TEST_MODE` | `true` | `require_ready()` refuses `false` |
| `RAZORPAY_TIMEOUT_SECONDS` | `15` | per request |
| `RAZORPAY_PAYMENT_LINK_EXPIRY_MINUTES` | `60` | clamped to ≥16 (Razorpay minimum is >15) |

Added as explicit typed optional fields on `app.config.Settings` — `extra="forbid"`
unchanged, no parallel settings mechanism. `GET /api/v1/recovery-events/{id}/razorpay`
reports a **secret-free** status snapshot.

## 3. Payment Link lifecycle

```
created ──(customer pays in Test Mode)──▶ paid   ──▶ webhook payment_link.paid   ──▶ Outcome(payment_recovered=true)
   │
   ├──(no payment, TTL reached)──▶ expired      ──▶ webhook payment_link.expired  ──▶ status only
   └──(cancelled)────────────────▶ cancelled    ──▶ webhook payment_link.cancelled──▶ status only
```

`expire_by` is set to `now + RAZORPAY_PAYMENT_LINK_EXPIRY_MINUTES` (epoch seconds).

## 4. Recovery event → intervention mapping

| Layer | Row |
|---|---|
| `RecoveryEvent` | the failed-payment recovery case (existing) |
| `Intervention` | one per agent action; now also carries `razorpay_reference_id` (UNIQUE), `razorpay_payment_link_id` (UNIQUE), `razorpay_short_url`, `razorpay_payment_id`, `last_razorpay_status` |
| Razorpay Payment Link | `reference_id = recovery-{recovery_event_id}-{intervention_id}`; `notes` carry both ids |
| Razorpay payment | `Intervention.razorpay_payment_id` (from the webhook) |
| `Outcome` | created **once**, only on verified `payment_link.paid`; `recovered_amount_paise` = Razorpay *payment* amount |

`Payment` continues to represent the **original failed payment**. On recovery its
`status` → `success` and `recovered_at` is set (matching the orchestrator's
existing `_is_already_recovered` semantics); the Razorpay recovery transaction id
lives on the `Intervention`, not by overwriting `Payment`.

## 5. Webhook verification

1. read the **raw** body (re-serialising would break the HMAC);
2. `signature = HMAC_SHA256(raw_body, RAZORPAY_WEBHOOK_SECRET)` compared to
   `X-Razorpay-Signature` with `hmac.compare_digest`;
3. **only then** parse JSON;
4. missing/invalid signature → HTTP 400, nothing touched.

## 6. Idempotency

* **API execution**: deterministic `reference_id` + `interventions.razorpay_payment_link_id`
  UNIQUE. Second call → reuse (one `POST`). Test:
  `tests/razorpay/test_execution_idempotency.py::test_second_call_reuses_the_existing_link`.
* **Webhook**: `processed_webhook_events` PK = `X-Razorpay-Event-Id`; plus the
  `Outcome.intervention_id` UNIQUE guard. Same body posted twice → one Outcome.
  Test: `tests/razorpay/test_webhooks.py::test_duplicate_delivery_by_event_id_is_idempotent`
  and `tests/razorpay/test_webhook_api.py::test_endpoint_processes_paid_and_is_idempotent`.

## 7. Outcome semantics

**Execution success ≠ payment recovered.** A created Payment Link never sets
`payment_recovered`. Only `app/services/razorpay_webhook.py`, on a
signature-verified `payment_link.paid`, creates the Outcome and marks recovery.
Tests: `test_execution_idempotency.py::test_link_creation_does_not_create_outcome_or_mark_recovered`,
`test_agent_razorpay.py::test_link_creation_does_not_mark_recovered`,
`test_webhooks.py::test_payment_link_paid_creates_single_outcome_and_closes_event`.

## 8. Dry-run vs real Test Mode

| | `dry_run=true` | `dry_run=false` |
|---|---|---|
| Razorpay calls | **none** | `POST /v1/payment_links` (idempotent) |
| DB writes | none | Intervention + correlation ids + AuditLog |
| tool result | `simulated:true`, no ids | real link id + `short_url` + `razorpay_status` |
| on failure | n/a | Intervention rolled back, structured error to Gemini |

Proof dry-run is inert: `test_agent_razorpay.py::test_dry_run_makes_no_razorpay_call`
asserts the fake transport was never invoked.

## 9. Agent loop

Unchanged. `execute_recovery_action` is still **not terminal** — after a link is
created the loop returns control to Gemini, which independently chooses
`observe_recovery_outcome` / another action / `stop_recovery` / `escalate_recovery`.
No `if link_created: auto-observe` anywhere. The non-determinism guard
(`test_structurally_different_events_produce_different_sequences`) still passes,
and `test_agent_razorpay.py::test_structurally_different_events_differ_with_razorpay`
shows a Razorpay-integrated run also branches on real tool results.

## 10. What is external vs simulated

**Real (code paths exist and are exercised against a real API contract):**
* Razorpay REST client — Basic auth, retries, error mapping (real client code, fake socket in tests)
* Webhook HMAC-SHA256 verification — real crypto, real secret
* DB correlation, Outcome creation, event closure, idempotency constraints

**Real Razorpay network calls: EXECUTED ONCE (Test Mode).** `scripts/razorpay_smoke.py`
was run against real `rzp_test_` credentials: it created one real Test Mode
Payment Link, the link was paid with a Razorpay test card, and the real
`payment_link.paid` webhook was delivered and signature-verified, producing the
Outcome for recovery event 18499. Repeat verification uses the test suite
(real client code + fake transport, real HMAC crypto) so the Test Mode Payment
Link limit is not consumed again.

**Not real (later stages):** live-mode (non-test) payments, real SMS, real
WhatsApp, outbound Hinglish voice delivery (a TTS audio *file* is generated; it
is not delivered to any customer).

## 11. Running the smoke test

```
# 1. put rzp_test_ credentials + a webhook secret in .env
# 2. start the API:   uvicorn app.main:app --port 8000
# 3. expose it:       ngrok http 8000     (or cloudflared tunnel --url http://localhost:8000)
# 4. in the Razorpay Dashboard, add a webhook -> https://<public>/api/v1/webhooks/razorpay
#    subscribe to: payment_link.paid, payment_link.cancelled, payment_link.expired
# 5. run:
.venv/Scripts/python.exe scripts/razorpay_smoke.py
#    -> dry-run (no side effects), then live: creates ONE real test link, prints its URL
# 6. open the URL, pay with a Razorpay test card (e.g. 4111 1111 1111 1111, any future expiry/CVV)
# 7. the webhook hits the tunnel -> Outcome is persisted
# 8. re-run:  scripts/razorpay_smoke.py --observe <recovery_event_id>
```

## 12. Known limitations

* **Real Razorpay Test Mode run: executed once** (recovery event 18499). It is
  not re-run in CI — repeated verification is via the mocked/locally-signed
  test suite — so the Test Mode Payment Link quota is not consumed on every run.
* Payment Link *notification* sending (`notify.sms/email`) is disabled — the
  synthetic customer emails are `@example.com` and phones are random. The
  generated `short_url` is returned for the caller to deliver.
* No real SMS/WhatsApp provider — `communication_sent` is always `false`.
* One Payment Link per intervention; `MAX_INTERVENTION_ATTEMPTS = 3` still caps
  attempts per recovery event.

## Sequence diagram

```
Failed Payment
    ↓
Recovery Event (open, non-control)
    ↓
Gemini Agent  ── chooses execute_recovery_action(action, hinglish msg, reason)
    ↓
Guardrails    ── event open? not control? not recovered? attempts<3? action eligible & untried?
    ↓ (pass)
RecoveryExecutionService ── amount/currency from Payment (server-side, NOT Gemini)
    ↓
Razorpay Payment Link (POST /v1/payment_links, reference_id = recovery-{re}-{iv})
    ↓                                   payment_recovered = FALSE
Intervention.razorpay_* persisted; tool returns link id + short_url to Gemini
    ↓
Gemini Agent  ── decides next turn: observe / another action / stop / escalate
    ↓
Customer opens short_url, pays with a Razorpay test card
    ↓
Razorpay Webhook  payment_link.paid  ──▶  POST /api/v1/webhooks/razorpay
    ↓
Signature verification (HMAC-SHA256 of raw body vs X-Razorpay-Signature)
    ↓ (valid)
Correlate by razorpay_payment_link_id → Intervention → RecoveryEvent
    ↓
Outcome(payment_recovered=TRUE, recovered_amount_paise = Razorpay payment amount)  [exactly once]
Payment.status = success ; RecoveryEvent.status = closed
processed_webhook_events row written (PK = X-Razorpay-Event-Id)
    ↓
observe_recovery_outcome  ──▶  payment_recovered = TRUE, amount, terminal = true
    ↓
Gemini Agent  ──▶  STOP (payment_recovered)   /   ESCALATE   /   NEXT ACTION
```
