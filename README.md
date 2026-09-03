# Razorpay AI Revenue Recovery

An agentic revenue-recovery system for failed subscription payments. A failed
charge isn't a lost customer — it's revenue at risk. This project **detects**
the failure, **understands** why it failed, **predicts** which recovery action
will work, lets a Gemini agent **decide** the next step, **recovers** the payment
via a Razorpay Test Mode Payment Link, and **verifies** the recovery with a
signature-checked webhook before ever calling it recovered.

```
Failed payment
  → Recovery event
  → Agent observes context
  → ML + uplift score every action
  → Agent picks the recovery action  (never invents amounts, never bypasses guardrails)
  → Deterministic guardrails validate it
  → Razorpay Test Mode Payment Link
  → Customer pays
  → Signed webhook → correlation → Outcome
  → Agent observes recovered state → stops
  → full audit trail
```

## Repository layout

| Path | What |
|---|---|
| [`backend/`](backend/) | FastAPI + SQLAlchemy + Alembic + Postgres. Recovery data model, rules / ML / uplift recovery policies, the agentic Gemini tool-calling loop, deterministic guardrails, Razorpay Test Mode integration + webhook verification, Hinglish TTS, and the analytics API. 232 passing tests. |
| [`frontend/`](frontend/) | Vite + React + TypeScript dashboard ("Reclaim"). Renders backend state only — it never decides recovery truth. Command-center overview, a recovery-event explorer, the signature per-event case file with the agent's turn-by-turn trace, analytics with a real Newcombe/Wilson confidence interval, experiments, and an audit trail. |

## Running it

**Backend** (needs Postgres + Python 3.14):

```bash
cd backend
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or bin/pip on POSIX
cp .env.example .env                     # fill in DATABASE_URL; Gemini / Razorpay keys are optional
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m app.scripts.generate_data --reset --customers 6000 --seed 42
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
cp .env.example .env.local               # VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev                               # http://localhost:5173
```

Optional demo data for the "Run agent" walkthrough:
`python -m scripts.seed_demo_events` (reversible with `--reset`).

## Notes

- **Test Mode only.** No live-money path exists. The Razorpay client refuses any
  non-`rzp_test_` key.
- The ML / uplift models are trained on **synthetic** data — a benchmark against a
  known generator, not evidence of a real causal effect. Every surface that shows
  a model number says so.
- No credentials are committed. `backend/.env` is gitignored; only `.env.example`
  templates are tracked.
