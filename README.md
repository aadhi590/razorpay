# Reclaim — AI Revenue Recovery

**An autonomous agent that reasons about which recoveries are worth pursuing, proves its impact with real controlled experiments instead of guessed numbers, and knows when to act and when to walk away.**

Built for Razorpay's AI Buildathon 2026 — Track 03: AI Revenue Recovery.

---

## The problem

Revenue loss rarely happens in one clean step. A payment degrades, a subscription fails, an intervention gets tried — and most systems either retry everything blindly (wasting compliance budget and annoying customers) or retry nothing systematically (leaving recoverable money on the table). Worse, almost no recovery system can answer the one question that actually matters: **of the money that came back, how much would have come back anyway, with no help at all?**

Reclaim is built around answering that question honestly, and acting on the answer.

---

## What makes this different

Most recovery tools optimize whether a payment *succeeds*. Reclaim optimizes whether **intervention actually caused the money to come back** — a materially harder, more honest claim — and it operates as a genuinely autonomous agent, not a scripted retry sequence.

- **Real causal measurement, not a reported number.** A genuine randomized control/treatment split is built into the data model itself. Reclaim doesn't just report a recovery rate — it reports a **95% confidence interval on the incremental lift**, computed with the Newcombe method, and states plainly whether that interval excludes zero (meaning the lift is unlikely to be sampling noise).
- **A genuinely agentic loop, not a fixed script.** The agent decides its own next step, turn by turn, based only on what it has actually observed so far. Two different recovery events produce two different reasoning paths and different numbers of turns — proven by a dedicated test, not just claimed.
- **Self-correcting reasoning.** Before trusting the model's predicted uplift for an action, the agent can consult the *actual observed* historical performance of that action — and reason honestly when the two disagree.
- **Scarcity-aware, not greedy.** A portfolio allocator ranks all currently-open recovery opportunities and enforces a real capacity constraint, showing exactly which events get acted on and which are skipped — and why — rather than treating every event as independent.
- **Operates on its own.** A bounded, capacity-respecting scheduler triggers recovery runs autonomously, with every autonomous action distinguishable from a manually-triggered one in the audit trail.

---

## Real, verified evidence — not a demo script

Every claim below was actually run and observed, not assumed:

| Claim | Evidence |
|---|---|
| Real Razorpay Test Mode payment recovered | Payment link `plink_TXIOWJC2UJRgcc` → real captured payment `pay_TXIcHbfpqUy35R` → signature-verified webhook → Outcome recorded, amount matched exactly (₹499.00) |
| Genuinely agentic, not scripted | Two structurally different recovery events produce two different tool-call sequences and different turn counts — asserted by a dedicated test |
| Real incremental measurement | Control 2.62% vs. treated 14.28% recovery rate → 95% CI **[10.6%, 12.6%]** on the incremental lift, excluding zero |
| Fails safely under real conditions | A live agent run hit a real Gemini quota limit mid-flow, after a real payment link had already been created — degraded safely, no state corruption, a second run correctly observed the recovered outcome and stopped |
| Portfolio allocation is real, not sorting | A lower-priority but larger, more-recoverable event correctly outranked a higher-priority one under a capacity constraint — the skip reasoning is as specific and auditable as the act reasoning |
| Provider-agnostic by design | The same agent, tools, and guardrails run identically behind Gemini or Groq, selected by one config flag — verified with a real live tool-calling run on each |

---

## Architecture

Reclaim separates concerns deliberately: an LLM reasons and orchestrates, quantitative models provide scores, deterministic guardrails hold the line, and the database is the only source of truth. The agent can propose; it can never fabricate a recovery, invent a number, or bypass a limit.

A high-level view of the full system — client, API, the orchestration and agent layer, the ML/uplift intelligence layer, external integrations, and the data layer:

<img width="1800" height="2360" alt="reclaim_tier1_architecture" src="https://github.com/user-attachments/assets/854ec1fa-b27d-4da4-9cb6-cb352ed8c918" />




### The agent's reasoning loop

The core differentiator in Reclaim is not any single model — it's the discipline of the loop itself. Each turn, the agent asks the active LLM provider for exactly one next tool call, based only on what it has observed so far. Every tool call is validated against a deterministic guardrail layer before it executes — eligibility, attempt limits, control-arm exclusion — none of which the model can see past or override. Read-only context and scoring tools inform the decision; a distinct cluster of self-check tools lets the agent compare the model's *predicted* effectiveness against what was *actually observed* historically, before committing. Executing an action is never treated as the end of the run — recovery is only ever confirmed by a signature-verified webhook, never assumed from execution alone.

**[ARCHITECTURE DIAGRAM PLACEHOLDER — Panel A: Agent Tool-Calling Loop]**

<img width="1800" height="2592" alt="panel_a_agent_loop" src="https://github.com/user-attachments/assets/324b2036-1658-4563-8ba1-bcd3327cabb4" />


### The intelligence layer

Two models feed the agent's decisions, both trained offline through a shared, leak-proof feature pipeline: a recovery-probability model predicting the likelihood an action succeeds, and a causal uplift model — trained against a genuine randomized control arm — predicting the *incremental* effect of an action versus doing nothing. The agent acts on the second number, not the first, because the whole point is to act only where intervention causes an outcome, not merely where one happens to be likely.

**[ARCHITECTURE DIAGRAM PLACEHOLDER — Panel B: ML & Causal Inference Pipelines]**

<img width="1800" height="2020" alt="panel_b_ml_pipeline (1)" src="https://github.com/user-attachments/assets/78c0a1c7-d864-4517-9016-76b03e9eec98" />


---

## Feature tour

### 1. The autonomous recovery agent
A bounded, tool-calling reasoning loop (Gemini or Groq, swappable behind one config flag) that observes a failed payment's context, consults real ML and uplift scores, cross-checks those scores against historical reality, executes at most one guardrail-cleared action, and decides for itself what to do next — observe, retry, escalate, or stop.

<img width="1366" height="692" alt="image" src="https://github.com/user-attachments/assets/4f953ab7-e866-4cfb-b84b-60bc876849b7" />


### 2. Real Razorpay Test Mode integration
When the agent chooses to act, it creates a real Razorpay Payment Link, idempotently tied to the specific intervention. A signature-verified webhook — never the agent itself — is the only thing that can mark a payment recovered. Execution success and payment recovery are treated as two different, non-interchangeable facts throughout the system.

### 3. Real incrementality measurement
A genuine randomized control/treatment split, a proper Newcombe 95% confidence interval, and an explicit statement of whether the observed lift is likely to be a real effect or sampling noise — the single most differentiated claim in the system.

<img width="1366" height="674" alt="image" src="https://github.com/user-attachments/assets/bb61cc44-eb27-48eb-a39c-cebbfe5449ca" />


<img width="1366" height="628" alt="image" src="https://github.com/user-attachments/assets/b932741f-616c-4925-8d7f-a80259ff7fbb" />



### 4. Historical incrementality self-check
A dedicated agent tool lets the model compare what the uplift model *predicted* for an action against what was *actually observed* across past interventions of that type — and say so, out loud, in its own reasoning.

### 5. Action lift trend detection
The same reality-check discipline applied across time: is a given action's real-world effectiveness improving, declining, or flat recently, compared to its all-time performance — reported honestly, including when the answer is "no detectable trend."

### 6. Portfolio scarcity allocator
Recovery capacity, not information, is treated as the scarce resource. Given a batch of simultaneously open events and a capacity limit, the allocator ranks all of them by expected incremental value and shows exactly who gets acted on and who is skipped — with a specific, auditable reason for every skip, not a generic "insufficient budget."
<img width="1365" height="684" alt="image" src="https://github.com/user-attachments/assets/d9e19ff3-1108-4535-a10d-6815ba25e4d1" />


### 7. Autonomous scheduler
A bounded, interval-driven scheduler that triggers real recovery runs on its own, respecting the same allocator ranking and the same guardrails as a manually triggered run — with every autonomous action distinguishable from a manual one in the audit trail.

### 8. Hinglish voice artifact generation
The agent's customer-facing message — generated in natural Hinglish — can also be synthesized into a real local audio file, honestly labeled as an artifact for retrieval, not a delivered phone call or message.

### 9. Full audit trail
Every decision, every tool call, every guardrail check, and every external confirmation is recorded — filterable and traceable end to end from a failed payment to a verified recovery.

---

## What's real vs. what's explicitly not yet

**Real:** Razorpay Test Mode payment links, signature-verified webhooks, outcome persistence, the full agentic loop on two interchangeable LLM providers, genuine randomized incrementality measurement, local Hinglish text-to-speech audio generation, autonomous scheduling.

**Not yet, and stated plainly:** live-mode payments, real SMS delivery, real WhatsApp delivery, real outbound voice calls. Where these would apply, the system generates the real underlying artifact (a link, a message, an audio file) and honestly reports that it was not delivered through any external channel — it never claims a message was sent when it wasn't.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python |
| Database | PostgreSQL, SQLAlchemy 2.x, Alembic |
| Frontend | React, TypeScript, Vite, TanStack Query |
| AI / LLM | Gemini and Groq — interchangeable via one config flag, stdlib REST, no framework |
| Payments | Razorpay Test Mode API (Payment Links, signed webhooks) |
| ML / Causal inference | scikit-learn, gradient-boosted uplift modeling |
| Voice | pyttsx3 (local text-to-speech, no network dependency) |

No LangChain, no second agent, no unnecessary orchestration layer — one disciplined, bounded, tool-calling loop, and a deterministic guardrail authority it cannot override.

---

## Running it locally

```bash
# backend
cd backend
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY or GROQ_API_KEY, RAZORPAY_* test-mode keys
uvicorn app.main:app --reload

# frontend
cd frontend
npm install
npm run dev
```

Generate synthetic data (customers, subscriptions, payments, failed payments, recovery events, interventions, outcomes, with a real control/treatment split):

```bash
python -m app.scripts.generate_data --customers 1000 --seed 42
```

---

## License

Built for the Razorpay AI Buildathon 2026.
