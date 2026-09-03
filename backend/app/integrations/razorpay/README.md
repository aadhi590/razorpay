# Razorpay integration (`app/integrations/razorpay/`)

**Test Mode only in this stage.** No live-money path exists. The client refuses
to run unless `RAZORPAY_TEST_MODE=true` and `RAZORPAY_KEY_ID` starts with
`rzp_test_`.

## What this package owns

| File | Responsibility |
|---|---|
| `config.py` | `RazorpayConfig` from `app.config.Settings`; `require_ready()` guard |
| `client.py` | stdlib-`urllib` REST client: Basic auth, HTTPS-only, timeout, bounded retries (5xx/429/network only), error mapping, credential redaction |
| `schemas.py` | `PaymentLinkCreateRequest`, `PaymentLink`, `WebhookEnvelope` (unknown fields ignored) |
| `webhooks.py` | `verify_signature()` — HMAC-SHA256(raw_body, webhook_secret) vs `X-Razorpay-Signature` |
| `exceptions.py` | typed error hierarchy; no message ever contains a secret |

No recovery business logic lives here. Call chain:

```
Gemini agent → execute_recovery_action tool → [guardrails]
    → RecoveryExecutionService (app/services/recovery_execution.py)
    → RazorpayClient (this package)
    → Razorpay Test Mode API
```

## Razorpay endpoints used

| Method | Path | When | Docs |
|---|---|---|---|
| `POST` | `/v1/payment_links` | create a Standard Payment Link for a recovery attempt | https://razorpay.com/docs/api/payments/payment-links/create-standard/ |
| `GET` | `/v1/payment_links/{id}` | refresh status on reuse | https://razorpay.com/docs/api/payments/payment-links/fetch-with-id/ |
| `GET` | `/v1/payment_links?reference_id=...` | recover from a duplicate-reference 400 | https://razorpay.com/docs/api/payments/payment-links/all/ |

`amount` is sent in the **smallest currency unit** (paise for INR) — it is copied
verbatim from `Payment.amount`, which is already stored in paise. Gemini never
supplies the amount or currency.

## Webhook events handled

| Event | Effect | Verified against |
|---|---|---|
| `payment_link.paid` | create the single `Outcome` (`payment_recovered=true`, amount from the Razorpay *payment* entity), mark `Payment` recovered, close the `RecoveryEvent` | https://razorpay.com/docs/webhooks/payloads/payment-links/ |
| `payment_link.partially_paid` | record `last_razorpay_status` only, no Outcome | same |
| `payment_link.cancelled` | record status only | same |
| `payment_link.expired` | record status only | same |
| anything else | 200, ignored | — |

## Idempotency

* **Payment Link creation** — deterministic `reference_id = recovery-{recovery_event_id}-{intervention_id}` + the `interventions.razorpay_payment_link_id` UNIQUE column. The service checks the column first and reuses; a Razorpay duplicate-reference 400 triggers a `GET ?reference_id=` fallback. Proven by `tests/razorpay/test_execution_idempotency.py::test_second_call_reuses_the_existing_link` (asserts one `POST`).
* **Webhook processing** — `processed_webhook_events` table, primary key = Razorpay's `X-Razorpay-Event-Id`. A duplicate delivery hits the PK and is acknowledged with no state change. Second guard: `Outcome.intervention_id` is UNIQUE and the handler checks for an existing Outcome. Proven by `tests/razorpay/test_webhooks.py::test_duplicate_delivery_by_event_id_is_idempotent` (posts the same body twice, asserts one Outcome).

## Outcome semantics (the invariant)

Creating a Payment Link is **execution success**, not payment recovery.
`payment_recovered` stays `false` until a signature-verified `payment_link.paid`
webhook is processed. Proven by
`tests/razorpay/test_execution_idempotency.py::test_link_creation_does_not_create_outcome_or_mark_recovered`
and `tests/razorpay/test_agent_razorpay.py::test_link_creation_does_not_mark_recovered`.

## Dry-run

`dry_run=true` → the tool returns a simulated result and makes **zero** calls.
Proven by `tests/razorpay/test_agent_razorpay.py::test_dry_run_makes_no_razorpay_call`
(asserts the fake transport was never invoked — fails if any real call is made).

## Config

See `.env.example`. `GET /api/v1/recovery-events/{id}/razorpay` returns a
secret-free config status snapshot plus this event's Razorpay state.
