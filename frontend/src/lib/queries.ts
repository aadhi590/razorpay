import {
  useQuery,
  useMutation,
  useQueryClient,
  keepPreviousData,
} from "@tanstack/react-query";
import { api } from "./api";
import { VERIFIED_RECOVERY_EVENT_ID } from "./config";
import type {
  ActionScoresResponse,
  ActionsResponse,
  AgentEventDetail,
  AgentEventListItem,
  AgentRunContext,
  AgentRunResult,
  AnalyticsSummary,
  AuditLogDetail,
  AuditLogListItem,
  ControlVsTreatment,
  Customer,
  ExperimentListItem,
  ExperimentsResponse,
  Health,
  InterventionListItem,
  ModelInfo,
  OutcomeListItem,
  Payment,
  PaymentListItem,
  PortfolioAllocation,
  RecoveryEvent,
  RecoveryEventListItem,
  RecoveryEventRazorpay,
  RecoveryImpact,
  Subscription,
  UpliftModelInfo,
  UpliftScoresResponse,
} from "./types";

const MIN = 60_000;

// ---- health ------------------------------------------------------------
export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: () => api.get<Health>("/health"),
    refetchInterval: 30_000,
    retry: 1,
  });

// ---- analytics -------------------------------------------------------
export const useSummary = () =>
  useQuery({
    queryKey: ["analytics", "summary"],
    queryFn: () => api.get<AnalyticsSummary>("/api/v1/analytics/summary"),
    staleTime: MIN,
  });

export const useControlVsTreatment = () =>
  useQuery({
    queryKey: ["analytics", "cvt"],
    queryFn: () => api.get<ControlVsTreatment>("/api/v1/analytics/control-vs-treatment"),
    staleTime: MIN,
  });

export const useActionAnalytics = () =>
  useQuery({
    queryKey: ["analytics", "actions"],
    queryFn: () => api.get<ActionsResponse>("/api/v1/analytics/actions"),
    staleTime: MIN,
  });

export const useExperimentAnalytics = () =>
  useQuery({
    queryKey: ["analytics", "experiments"],
    queryFn: () => api.get<ExperimentsResponse>("/api/v1/analytics/experiments"),
    staleTime: MIN,
  });

export const useRecoveryImpact = () =>
  useQuery({
    queryKey: ["analytics", "impact"],
    queryFn: () => api.get<RecoveryImpact>("/api/v1/analytics/recovery-impact"),
    staleTime: MIN,
  });

export const usePortfolioAllocation = (capacity: number) =>
  useQuery({
    queryKey: ["analytics", "portfolio-allocation", capacity],
    queryFn: () =>
      api.get<PortfolioAllocation>(
        `/api/v1/analytics/portfolio-allocation?capacity=${capacity}`,
      ),
    staleTime: MIN,
    placeholderData: keepPreviousData,
  });

// ---- model cards ----------------------------------------------------
export const useMlModel = () =>
  useQuery({
    queryKey: ["model", "ml"],
    queryFn: () => api.get<ModelInfo>("/api/v1/ml/model"),
    staleTime: 10 * MIN,
  });
export const useUpliftModel = () =>
  useQuery({
    queryKey: ["model", "uplift"],
    queryFn: () => api.get<UpliftModelInfo>("/api/v1/uplift/model"),
    staleTime: 10 * MIN,
  });

// ---- big lists (fetched once, cached hard) --------------------------
const bigList = { staleTime: 5 * MIN, gcTime: 30 * MIN } as const;

export const useRecoveryEventList = () =>
  useQuery({
    queryKey: ["list", "recovery-events"],
    queryFn: () => api.get<RecoveryEventListItem[]>("/api/v1/recovery-events/"),
    ...bigList,
  });
export const usePaymentList = () =>
  useQuery({
    queryKey: ["list", "payments"],
    queryFn: () => api.get<PaymentListItem[]>("/api/v1/payments/"),
    ...bigList,
  });
export const useInterventionList = () =>
  useQuery({
    queryKey: ["list", "interventions"],
    queryFn: () => api.get<InterventionListItem[]>("/api/v1/interventions/"),
    ...bigList,
  });
export const useOutcomeList = () =>
  useQuery({
    queryKey: ["list", "outcomes"],
    queryFn: () => api.get<OutcomeListItem[]>("/api/v1/outcomes/"),
    ...bigList,
  });
export const useAuditLogList = () =>
  useQuery({
    queryKey: ["list", "audit-logs"],
    queryFn: () => api.get<AuditLogListItem[]>("/api/v1/audit-logs/"),
    ...bigList,
  });
export const useAgentEventList = () =>
  useQuery({
    queryKey: ["list", "agent-events"],
    queryFn: () => api.get<AgentEventListItem[]>("/api/v1/agent-events/"),
    ...bigList,
  });
export const useExperimentList = () =>
  useQuery({
    queryKey: ["list", "experiments"],
    queryFn: () => api.get<ExperimentListItem[]>("/api/v1/experiments/"),
    staleTime: 10 * MIN,
  });

// ---- per-record ---------------------------------------------------
export const useRecoveryEvent = (id: number | null) =>
  useQuery({
    queryKey: ["recovery-event", id],
    queryFn: () => api.get<RecoveryEvent>(`/api/v1/recovery-events/${id}`),
    enabled: id != null,
  });
export const useRecoveryRazorpay = (id: number | null) =>
  useQuery({
    queryKey: ["recovery-event", id, "razorpay"],
    queryFn: () => api.get<RecoveryEventRazorpay>(`/api/v1/recovery-events/${id}/razorpay`),
    enabled: id != null,
  });
export const usePayment = (id: number | null) =>
  useQuery({
    queryKey: ["payment", id],
    queryFn: () => api.get<Payment>(`/api/v1/payments/${id}`),
    enabled: id != null,
  });
export const useSubscription = (id: number | null) =>
  useQuery({
    queryKey: ["subscription", id],
    queryFn: () => api.get<Subscription>(`/api/v1/subscriptions/${id}`),
    enabled: id != null,
  });
export const useCustomer = (id: number | null) =>
  useQuery({
    queryKey: ["customer", id],
    queryFn: () => api.get<Customer>(`/api/v1/customers/${id}`),
    enabled: id != null,
  });
export const useAgentEvent = (id: number | null) =>
  useQuery({
    queryKey: ["agent-event", id],
    queryFn: () => api.get<AgentEventDetail>(`/api/v1/agent-events/${id}`),
    enabled: id != null,
  });
export const useAuditLog = (id: number | null) =>
  useQuery({
    queryKey: ["audit-log", id],
    queryFn: () => api.get<AuditLogDetail>(`/api/v1/audit-logs/${id}`),
    enabled: id != null,
  });

export const useActionScores = (id: number | null) =>
  useQuery({
    queryKey: ["scores", "ml", id],
    queryFn: () => api.get<ActionScoresResponse>(`/api/v1/ml/recovery-events/${id}/action-scores`),
    enabled: id != null,
    staleTime: 5 * MIN,
  });
export const useUpliftScores = (id: number | null) =>
  useQuery({
    queryKey: ["scores", "uplift", id],
    queryFn: () => api.get<UpliftScoresResponse>(`/api/v1/uplift/recovery-events/${id}/uplift-scores`),
    enabled: id != null,
    staleTime: 5 * MIN,
  });

// ---- derived: the persisted agent run for one recovery event --------
export interface EventAgentRun {
  agentEventId: number;
  detail: AgentEventDetail;
  ctx: AgentRunContext | null;
}

/** A run that reached a real decision (executed / escalated / clean stop) ranks
 *  above one that only degraded safely (e.g. a rate-limited stop). Ties break to
 *  the most recent. */
function runQuality(item: AgentEventListItem): number {
  const d = item.decision ?? "";
  if (d.startsWith("execute:") || d.startsWith("escalate:")) return 3;
  if (d.startsWith("stop:") && !d.includes("quota") && !d.includes("failure")) return 2;
  if (item.confidence != null) return 1;
  return 0;
}
function bestRun(items: AgentEventListItem[]): AgentEventListItem | undefined {
  return [...items].sort((a, b) => runQuality(b) - runQuality(a) || b.id - a.id)[0];
}

export function useEventAgentRun(recoveryEventId: number | null) {
  const list = useAgentEventList();
  const runItem =
    recoveryEventId != null && list.data
      ? bestRun(
          list.data.filter(
            (e) =>
              e.recovery_event_id === recoveryEventId &&
              e.event_type === "agent_recovery_run",
          ),
        )
      : undefined;

  const detailQ = useAgentEvent(runItem?.id ?? null);

  const ctx =
    detailQ.data?.input_context &&
    typeof detailQ.data.input_context === "object" &&
    "tool_trace" in detailQ.data.input_context
      ? (detailQ.data.input_context as AgentRunContext)
      : null;

  return {
    isLoading: list.isLoading || detailQ.isLoading,
    isError: list.isError || detailQ.isError,
    error: list.error || detailQ.error,
    found: !!runItem,
    data:
      runItem && detailQ.data
        ? ({ agentEventId: runItem.id, detail: detailQ.data, ctx } as EventAgentRun)
        : undefined,
  };
}

/** Recovery events that have a recorded agent run — the ones worth showcasing.
 *  One row per recovery event (its best-quality run), newest event first. */
export function useAgentRunEvents() {
  const list = useAgentEventList();
  const byEvent = new Map<number, AgentEventListItem[]>();
  for (const e of list.data ?? []) {
    if (e.event_type !== "agent_recovery_run") continue;
    const arr = byEvent.get(e.recovery_event_id) ?? [];
    arr.push(e);
    byEvent.set(e.recovery_event_id, arr);
  }
  const unique = [...byEvent.values()]
    .map((items) => bestRun(items)!)
    .sort((a, b) => b.recovery_event_id - a.recovery_event_id);
  return { ...list, runs: unique };
}

// ---- the recovery index (join for the explorer) --------------------
export interface RecoveryRow {
  id: number;
  status: string;
  priority: number;
  created_at: string;
  amount_paise: number;
  currency: string;
  payment_status: string;
  recovered: boolean;
  interventionCount: number;
  actions: string[];
  hasAgentRun: boolean;
  agentAction: string | null;
  isControl: boolean | null;
}

export function useRecoveryIndex() {
  const events = useRecoveryEventList();
  const payments = usePaymentList();
  const interventions = useInterventionList();
  const outcomes = useOutcomeList();
  const agentRuns = useAgentEventList();

  const isLoading =
    events.isLoading ||
    payments.isLoading ||
    interventions.isLoading ||
    outcomes.isLoading;
  const isError =
    events.isError || payments.isError || interventions.isError || outcomes.isError;
  const error = events.error || payments.error || interventions.error || outcomes.error;

  let rows: RecoveryRow[] = [];
  if (events.data && payments.data && interventions.data && outcomes.data) {
    const payById = new Map(payments.data.map((p) => [p.id, p]));
    const recoveredIvIds = new Set(
      outcomes.data.filter((o) => o.payment_recovered).map((o) => o.intervention_id),
    );
    const ivByEvent = new Map<number, InterventionListItem[]>();
    for (const iv of interventions.data) {
      const arr = ivByEvent.get(iv.recovery_event_id) ?? [];
      arr.push(iv);
      ivByEvent.set(iv.recovery_event_id, arr);
    }
    const agentDecisionByEvent = new Map<number, string>();
    for (const e of agentRuns.data ?? []) {
      if (e.event_type !== "agent_recovery_run" || !e.decision) continue;
      const cur = agentDecisionByEvent.get(e.recovery_event_id);
      // an execute/escalate decision beats a bare stop
      if (!cur || (/^(execute|escalate):/.test(e.decision) && !/^(execute|escalate):/.test(cur))) {
        agentDecisionByEvent.set(e.recovery_event_id, e.decision);
      }
    }

    rows = events.data.map((ev) => {
      const p = payById.get(ev.payment_id);
      const ivs = ivByEvent.get(ev.id) ?? [];
      const recovered =
        p?.status === "success" || ivs.some((iv) => recoveredIvIds.has(iv.id));
      const decision = agentDecisionByEvent.get(ev.id) ?? null;
      const agentAction = decision?.startsWith("execute:")
        ? decision.slice("execute:".length)
        : null;
      return {
        id: ev.id,
        status: ev.status,
        priority: ev.priority,
        created_at: ev.created_at,
        amount_paise: p?.amount ?? 0,
        currency: p?.currency ?? "INR",
        payment_status: p?.status ?? "unknown",
        recovered: !!recovered,
        interventionCount: ivs.length,
        actions: [...new Set(ivs.map((iv) => iv.action_type))],
        hasAgentRun: agentDecisionByEvent.has(ev.id),
        agentAction,
        isControl: null,
      };
    });
    rows.sort((a, b) => b.id - a.id);
  }

  return { rows, isLoading, isError, error };
}

// ---- agent run mutation ------------------------------------------
export function useRunAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, dryRun }: { id: number; dryRun: boolean }) =>
      api.post<AgentRunResult>(
        `/api/v1/agent/recovery-events/${id}/run?dry_run=${dryRun ? "true" : "false"}`,
      ),
    onSuccess: (_data, { id }) => {
      qc.invalidateQueries({ queryKey: ["list", "agent-events"] });
      qc.invalidateQueries({ queryKey: ["recovery-event", id] });
    },
  });
}

export { keepPreviousData, VERIFIED_RECOVERY_EVENT_ID };
