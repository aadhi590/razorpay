/**
 * Maps the agent's real tool names (app/agent/tools/*) to concise, human-readable
 * phase labels for the timeline. These are presentation labels for tools the
 * backend actually ran — not a fabricated script.
 */
export interface Phase {
  label: string;
  description: string;
  kind: "observe" | "intelligence" | "decide" | "execute" | "verify" | "stop" | "escalate";
}

export const TOOL_PHASES: Record<string, Phase> = {
  get_recovery_event_context: {
    label: "Observe",
    description: "Load the failed-payment recovery context",
    kind: "observe",
  },
  get_payment_context: {
    label: "Observe",
    description: "Inspect the failed payment: amount, reason, timing",
    kind: "observe",
  },
  get_subscription_context: {
    label: "Observe",
    description: "Read the subscription behind the payment",
    kind: "observe",
  },
  get_customer_recovery_history: {
    label: "Analyze",
    description: "Review the customer's payment reliability and past recoveries",
    kind: "observe",
  },
  get_available_recovery_actions: {
    label: "Options",
    description: "Determine which recovery actions are eligible right now",
    kind: "observe",
  },
  get_action_scores: {
    label: "Intelligence",
    description: "Score every eligible action with the ML and uplift models",
    kind: "intelligence",
  },
  get_historical_incrementality_for_action: {
    label: "Reality check",
    description:
      "Compare the model's predicted uplift against the action's real, measured historical lift",
    kind: "intelligence",
  },
  get_action_lift_trend: {
    label: "Trend check",
    description:
      "Check whether the action's real-world effectiveness is trending — recent window vs. earlier",
    kind: "intelligence",
  },
  execute_recovery_action: {
    label: "Execute",
    description: "Run the chosen action — create a Razorpay recovery Payment Link",
    kind: "execute",
  },
  observe_recovery_outcome: {
    label: "Verify",
    description: "Check the real recovery state — is the payment confirmed?",
    kind: "verify",
  },
  stop_recovery: {
    label: "Stop",
    description: "End the run with a decision rationale",
    kind: "stop",
  },
  escalate_recovery: {
    label: "Escalate",
    description: "Hand the case to a human / team",
    kind: "escalate",
  },
};

export function phaseFor(tool: string): Phase {
  return (
    TOOL_PHASES[tool] ?? {
      label: tool.replace(/_/g, " "),
      description: "",
      kind: "observe",
    }
  );
}

export const STOP_REASON_LABELS: Record<string, string> = {
  payment_recovered: "Payment recovered",
  no_eligible_actions: "No eligible actions left",
  max_attempts_reached: "Attempt cap reached",
  expected_value_below_threshold: "Expected value too low",
  customer_already_recovered: "Customer already recovered",
  control_event: "Control event — no intervention",
  action_executed_awaiting_outcome: "Action executed — awaiting payment",
  guardrail_violation: "Blocked by guardrail",
  quota_or_api_failure: "AI provider unavailable",
  max_turns_reached: "Reasoning turn limit reached",
  repeated_invalid_output: "Repeated invalid model output",
  escalation_required: "Escalated to a human",
  other: "Run ended",
};
export function stopReasonLabel(r: string | null | undefined): string {
  if (!r) return "—";
  return STOP_REASON_LABELS[r] ?? r.replace(/_/g, " ");
}
