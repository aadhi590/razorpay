# Reclaim — AI Revenue Recovery (frontend)

A dashboard for the Razorpay AI Recovery Orchestrator. It renders backend state
only — it never decides recovery truth, amounts, allowed actions, agent
decisions, or payment success. Every figure on screen comes from an API
response.

## Stack

Vite 5 · React 18 · TypeScript (strict) · Tailwind CSS 3 (custom token layer) ·
TanStack Query · React Router 6 · Recharts · Lucide. No component library — the
UI is a bespoke design system (`src/index.css` tokens + `tailwind.config.js`).

## Run it

```bash
# 1. Start the backend (from ../backend)
#    .venv/Scripts/python -m uvicorn app.main:app --port 8000
#
# 2. Point the frontend at it
cp .env.example .env.local          # VITE_API_BASE_URL=http://127.0.0.1:8000
#
# 3. Dev server
npm install
npm run dev                         # http://localhost:5173
#
# or a production build
npm run build && npm run preview    # http://localhost:4173
```

The backend must allow the frontend's origin via `CORS_ALLOW_ORIGINS` (the
default already lists `localhost:5173` and `localhost:4173`). If
`VITE_API_BASE_URL` is left unset, the app calls a same-origin `/api` path and
the Vite dev proxy forwards it to `http://127.0.0.1:8000`.

## Scripts

| command | what it does |
|---|---|
| `npm run dev` | dev server with HMR |
| `npm run build` | typecheck (`tsc -b`) then production bundle to `dist/` |
| `npm run preview` | serve the production bundle |
| `npm run lint` | ESLint (flat config, typescript-eslint) |
| `npm run typecheck` | `tsc -b --noEmit` |

## Pages

- **Overview** — hero KPIs (revenue recovered / at risk / recovery rate), the
  DETECT → … → STOP narrative, causal-lift chart, recorded agent decisions
  (verified recovery pinned first), intervention performance, recent audit.
- **Recoveries** — client-joined explorer over the recovery-event / payment /
  intervention / outcome lists, with status + action + agent filters and
  client-side pagination.
- **Recovery detail** — the case file: recovery journey, the agent's recorded
  tool-by-tool trace, its decision + rationale, ML/uplift intervention
  intelligence, the Razorpay recovery card, the Hinglish customer message + voice
  artifact, and the evidence trail. A "Run live" button triggers a real Gemini
  dry run (rate-limited on the free tier; degrades to a designed safe state).
- **Analytics** — incremental revenue with a 95% CI, control-vs-treatment, per
  action performance, and the ML + uplift model cards (calibration, Qini, stated
  limitations).
- **Experiments** — per-experiment control/treatment lift.
- **Audit** — the verified recovery's end-to-end evidence, plus the full system
  activity log.

## Security

The only configuration the frontend takes is `VITE_API_BASE_URL`, a public URL.
No API keys, webhook secrets, or credentials are read, stored, or sent. All
requests are read-only GETs except one `POST` that triggers a dry-run agent
execution.
