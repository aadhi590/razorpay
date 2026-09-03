/**
 * Typed mirrors of the backend's response schemas (app/schemas/*, app/agent/schemas.py).
 * These are READ models only — the frontend never constructs recovery/agent/payment
 * truth, it only renders what these endpoints return.
 *
 * Monetary fields: the analytics endpoints return Decimal *rupee* strings; the
 * recovery-impact and per-event endpoints return integer *paise*. Field names
 * keep the backend's `_paise` / value distinction so the UI formats correctly.
 */

// ---- analytics ------------------------------------------------------------
export interface AnalyticsSummary {
  total_failed_payments: number;
  total_recovery_events: number;
  open_events: number;
  closed_events: number;
  abandoned_events: number;
  recovered_events: number;
  total_recovered_value: string; // rupees
  total_failed_payment_value: string; // rupees
  overall_recovery_rate: number; // 0..1
  average_payment_amount: string; // rupees
  total_interventions: number;
  total_outcomes: number;
}

export interface GroupStats {
  group: string;
  recovery_events: number;
  recovered_events: number;
  recovery_rate: number;
  total_payment_value: string;
  recovered_value: string;
}

export interface ControlVsTreatment {
  control: GroupStats;
  treatment: GroupStats;
  absolute_lift: number;
  relative_lift: number | null;
}

export interface ActionStats {
  action_type: string;
  interventions: number;
  distinct_recovery_events: number;
  recovered_events: number;
  recovery_rate: number;
  recovered_value: string;
  intervention_cost: string;
  cost_per_recovery: string | null;
}
export interface ActionsResponse {
  actions: ActionStats[];
}

export interface ExperimentVariantStats {
  experiment_id: number | null;
  experiment_name: string | null;
  variant: string | null;
  recovery_events: number;
  recovered_events: number;
  recovery_rate: number;
  payment_value: string;
  recovered_value: string;
}
export interface ExperimentResult {
  experiment_id: number | null;
  experiment_name: string | null;
  variants: ExperimentVariantStats[];
  absolute_lift: number | null;
  relative_lift: number | null;
}
export interface ExperimentsResponse {
  experiments: ExperimentResult[];
}

export interface RecoveryImpact {
  computable: boolean;
  reason: string | null;
  filters: { since: string | null; experiment_id: number | null };
  control_group_size: number;
  treated_group_size: number;
  recovered_control_events: number;
  recovered_treated_events: number;
  control_recovery_rate: number | null;
  treated_recovery_rate: number | null;
  incremental_recovery_rate: number | null;
  incremental_recovery_rate_ci_95: [number, number] | null;
  control_at_risk_amount_paise: number;
  treated_at_risk_amount_paise: number;
  control_recovered_revenue_paise: number;
  treated_recovered_revenue_paise: number;
  total_recovered_revenue_paise: number;
  incremental_revenue_recovered_paise: number | null;
  confidence_note: string;
  confidence_method: string;
}

// ---- core records -------------------------------------------------------
export interface RecoveryEventListItem {
  id: number;
  payment_id: number;
  status: string; // open | closed | abandoned
  priority: number;
  created_at: string;
}
export interface RecoveryEvent extends RecoveryEventListItem {
  closed_at: string | null;
}

export interface PaymentListItem {
  id: number;
  subscription_id: number;
  amount: number; // paise
  currency: string;
  status: string; // failed | success
  retry_count: number;
}
export interface Payment extends PaymentListItem {
  failure_reason: string | null;
  failed_at: string | null;
  recovered_at: string | null;
}

export interface InterventionListItem {
  id: number;
  recovery_event_id: number;
  action_type: string;
  status: string;
  cost_paise: number;
}
export interface OutcomeListItem {
  id: number;
  intervention_id: number;
  payment_recovered: boolean;
  recovered_amount_paise: number;
}

export interface Customer {
  id: number;
  external_customer_id: string;
  email: string | null;
  phone: string | null;
  total_successful_payments: number;
  total_failed_payments: number;
  created_at?: string;
}
export interface Subscription {
  id: number;
  customer_id: number;
  external_subscription_id: string;
  amount: number;
  currency: string;
  status: string;
  started_at?: string;
  next_payment_at?: string | null;
}

// ---- per-event Razorpay view ------------------------------------------
export interface RazorpayInterventionView {
  intervention_id: number;
  action_type: string;
  status: string;
  razorpay_payment_link_id: string | null;
  razorpay_short_url: string | null;
  razorpay_reference_id: string | null;
  razorpay_payment_id: string | null;
  last_razorpay_status: string | null;
  payment_link_created: boolean;
  payment_link_paid: boolean;
  outcome_recorded: boolean;
  outcome_payment_recovered: boolean;
  recovered_amount_paise: number | null;
}
export interface RazorpayConfigStatus {
  test_mode: boolean;
  base_url: string;
  key_id_configured: boolean;
  key_id_is_test_key: boolean;
  webhook_secret_configured: boolean;
  ready_for_live_calls: boolean;
}
export interface RecoveryEventRazorpay {
  recovery_event_id: number;
  status: string;
  is_control: boolean;
  payment_recovered: boolean;
  payment_status: string;
  amount_paise: number;
  interventions: RazorpayInterventionView[];
  razorpay_config: RazorpayConfigStatus;
}

// ---- ML / uplift -----------------------------------------------------
export interface ActionScore {
  action: string;
  probability: number;
  cost_paise: number;
  expected_value_paise: number;
}
export interface ActionScoresResponse {
  recovery_event_id: number;
  model_available: boolean;
  model_version: string | null;
  as_of: string;
  untried_actions: string[];
  scores: ActionScore[];
  recommended_action: string | null;
  note: string;
}
export interface ActionUplift {
  action: string;
  treatment_probability: number;
  uplift: number;
  cost_paise: number;
  incremental_expected_revenue_paise: number;
  net_incremental_value_paise: number;
  rank: number;
}
export interface UpliftScoresResponse {
  recovery_event_id: number;
  available: boolean;
  model_version: string | null;
  as_of: string;
  baseline_probability: number | null;
  amount_paise: number | null;
  untried_actions: string[];
  actions: ActionUplift[];
  recommended_action: string | null;
  note: string;
}
export interface ModelInfo {
  available: boolean;
  model_name?: string | null;
  model_version?: string | null;
  algorithm?: string | null;
  feature_version?: string | null;
  created_at?: string | null;
  selected_reason?: string | null;
  dataset?: Record<string, unknown> | null;
  test_metrics?: Record<string, unknown> | null;
  calibration?: Record<string, unknown> | null;
  synthetic_benchmark: boolean;
  detail?: string | null;
}
export interface UpliftModelInfo {
  available: boolean;
  model_name?: string | null;
  model_version?: string | null;
  learner_type?: string | null;
  base_algorithm?: string | null;
  champion_reason?: string | null;
  dataset?: Record<string, unknown> | null;
  propensity_diagnostics?: Record<string, unknown> | null;
  test_evaluation?: Record<string, unknown> | null;
  limitations?: string[] | null;
  synthetic_benchmark: boolean;
  detail?: string | null;
}

// ---- agent ----------------------------------------------------------
export interface ToolTraceEntry {
  turn: number;
  tool: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  terminal: boolean;
  result_summary: string;
  guardrail_code: string | null;
  latency_ms: number | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
}
export interface QuantScore {
  action: string;
  eligible?: boolean;
  cost_paise: number;
  recovery_probability?: number;
  expected_value_paise?: number;
  uplift?: number;
  treatment_probability?: number;
  net_incremental_value_paise?: number;
}

/**
 * One entry of AgentRunResult.action_incrementality — the observed, measured
 * historical incremental lift the agent looked up for an action type via
 * get_historical_incrementality_for_action. Mirrors
 * app/schemas/analytics.py::ActionIncrementalityResponse (compact tool payload).
 */
export interface ActionIncrementality {
  action_type: string;
  computable: boolean;
  reason?: string | null;
  treated_group_size: number;
  control_group_size: number;
  observed_recovery_rate_for_action?: number;
  baseline_control_recovery_rate?: number;
  observed_incremental_lift?: number;
  observed_incremental_lift_ci_95?: [number, number];
  /** the uplift model's prediction for THIS event, echoed from get_action_scores
   *  if that tool was called this run; absent otherwise */
  model_predicted_uplift_for_context?: number;
  note: string;
}
export interface AgentRunResult {
  recovery_event_id: number;
  agent: "gemini";
  model: string;
  dry_run: boolean;
  status: "completed" | "escalated" | "failed_safe";
  stop_reason: string;
  decision: string;
  chosen_action: string | null;
  customer_message: string | null;
  reasoning_summary: string;
  escalation_required: boolean;
  escalation_type: string | null;
  voice_generated: boolean;
  voice_reason: string | null;
  audio_url: string | null;
  voice_engine: string | null;
  turns_used: number;
  actions_attempted: string[];
  actions_executed: string[];
  outcome: Record<string, unknown> | null;
  latency_ms: number;
  token_usage: { prompt_tokens?: number; output_tokens?: number; total_tokens?: number };
  quantitative_scores: QuantScore[] | null;
  action_incrementality: Record<string, ActionIncrementality> | null;
  tool_trace: ToolTraceEntry[];
  errors: string[];
}

export interface AgentEventListItem {
  id: number;
  recovery_event_id: number;
  event_type: string;
  decision: string | null;
  confidence: number | null;
}
export interface AgentEventDetail extends AgentEventListItem {
  input_context: AgentRunContext | Record<string, unknown> | null;
  created_at: string;
}
/** The persisted shape written by app/agent/runner.py for event_type=agent_recovery_run. */
export interface AgentRunContext {
  agent_run_id: string;
  agent: string;
  model: string;
  dry_run: boolean;
  status: string;
  stop_reason: string;
  decision: string;
  chosen_action: string | null;
  escalation_required: boolean;
  escalation_type: string | null;
  turns_used: number;
  latency_ms: number;
  token_usage: { prompt_tokens?: number; output_tokens?: number; total_tokens?: number };
  actions_attempted: string[];
  actions_executed: string[];
  reasoning_summary: string;
  voice?: {
    voice_generated: boolean;
    voice_reason: string | null;
    audio_url: string | null;
    voice_engine: string | null;
    note: string;
  };
  quantitative_scores: QuantScore[] | null;
  action_incrementality?: Record<string, ActionIncrementality> | null;
  tool_trace: ToolTraceEntry[];
  tools_requested_in_order: string[];
  outcome: Record<string, unknown> | null;
  errors: string[];
}

// ---- audit --------------------------------------------------------
export interface AuditLogListItem {
  id: number;
  actor: string;
  action: string;
  created_at: string;
}
export interface AuditLogDetail {
  id: number;
  recovery_event_id: number | null;
  actor: string;
  action: string;
  reason: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface ExperimentListItem {
  id: number;
  name: string;
  intervention_type: string;
  status: string;
}

export interface Health {
  status: string;
  database: number;
}
