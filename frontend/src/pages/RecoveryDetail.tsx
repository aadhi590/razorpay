import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ShieldCheck } from "lucide-react";
import { Page } from "@/components/layout/Shell";
import { ApiError } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Skeleton, SkeletonText } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/States";
import { RecoveryJourney, type Stage } from "@/components/recovery/RecoveryJourney";
import { RazorpayCard } from "@/components/recovery/RazorpayCard";
import { CustomerMessageCard } from "@/components/recovery/CustomerMessageCard";
import { EvidenceTrail } from "@/components/recovery/EvidenceTrail";
import { InterventionIntelligence } from "@/components/recovery/InterventionIntelligence";
import { AgentRunPanel } from "@/components/agent/AgentRunPanel";
import {
  useRecoveryEvent,
  useRecoveryRazorpay,
  usePayment,
  useSubscription,
  useCustomer,
  useEventAgentRun,
  useActionScores,
  useUpliftScores,
  useAgentEventList,
} from "@/lib/queries";
import { VERIFIED_RECOVERY_EVENT_ID } from "@/lib/config";
import type { QuantScore } from "@/lib/types";
import {
  actionLabel,
  dateTime,
  failureLabel,
  relTime,
  rupeesFromPaise,
} from "@/lib/format";

function useMergedScores(
  eventId: number,
  agentScores: QuantScore[] | null | undefined,
  agentRunResolved: boolean,
): { scores: QuantScore[]; source: string | null } {
  // Only hit the (point-in-time SQL) score endpoints once we know the recorded
  // agent run has no scores of its own — otherwise we'd fire them needlessly.
  const needLive = agentRunResolved && !agentScores?.length;
  const ml = useActionScores(needLive ? eventId : null);
  const uplift = useUpliftScores(needLive ? eventId : null);

  return useMemo(() => {
    if (agentScores && agentScores.length) {
      return { scores: agentScores, source: "ml+uplift (recorded)" };
    }
    const byAction = new Map<string, QuantScore>();
    for (const s of ml.data?.scores ?? []) {
      byAction.set(s.action, {
        action: s.action,
        cost_paise: s.cost_paise,
        recovery_probability: s.probability,
        expected_value_paise: s.expected_value_paise,
      });
    }
    for (const a of uplift.data?.actions ?? []) {
      const prev = byAction.get(a.action) ?? { action: a.action, cost_paise: a.cost_paise };
      byAction.set(a.action, {
        ...prev,
        uplift: a.uplift,
        treatment_probability: a.treatment_probability,
        net_incremental_value_paise: a.net_incremental_value_paise,
      });
    }
    const parts = [ml.data?.model_available ? "ml" : null, uplift.data?.available ? "uplift" : null].filter(
      Boolean,
    );
    return {
      scores: [...byAction.values()],
      source: parts.length ? parts.join("+") : null,
    };
  }, [agentScores, ml.data, uplift.data]);
}

function buildStages(args: {
  hasAgentRun: boolean;
  chosenAction: string | null;
  interventionExecuted: boolean;
  linkCreated: boolean;
  linkPaid: boolean;
  recovered: boolean;
  eventStatus: string;
  amountLabel: string;
  isControl: boolean;
}): Stage[] {
  const {
    hasAgentRun,
    chosenAction,
    interventionExecuted,
    linkCreated,
    linkPaid,
    recovered,
    eventStatus,
    amountLabel,
    isControl,
  } = args;

  const done = (b: boolean): Stage["state"] => (b ? "done" : "pending");

  return [
    {
      key: "failed",
      title: "Payment failed",
      detail: "A subscription charge was declined and entered the recovery engine.",
      state: "done",
    },
    {
      key: "at-risk",
      title: `${amountLabel} at risk`,
      detail: "Revenue flagged for recovery — not written off.",
      state: "done",
    },
    isControl
      ? {
          key: "control",
          title: "Control event",
          detail: "Held out of intervention to measure the AI's true lift. No action taken.",
          state: "skipped",
        }
      : {
          key: "analysis",
          title: "AI analysis",
          detail: hasAgentRun
            ? "The agent observed the context and scored every eligible action."
            : "Not yet analysed by the agent.",
          state: done(hasAgentRun),
        },
    {
      key: "decision",
      title: chosenAction ? `Decision — ${actionLabel(chosenAction)}` : "Decision",
      detail: chosenAction
        ? "The agent selected this action from the scored options."
        : "No action selected yet.",
      state: done(!!chosenAction),
      tone: "accent",
    },
    {
      key: "intervention",
      title: "Intervention executed",
      detail: interventionExecuted
        ? "The recovery action was carried out."
        : "Pending.",
      state: done(interventionExecuted),
    },
    {
      key: "razorpay",
      title: "Razorpay Payment Link created",
      detail: linkCreated
        ? "A Test Mode Payment Link was generated with the authoritative amount."
        : "This recovery used a direct intervention — no Payment Link.",
      state: linkCreated ? "done" : interventionExecuted ? "skipped" : "pending",
    },
    {
      key: "confirmed",
      title: "Payment confirmed",
      detail: linkPaid
        ? "A signature-verified webhook confirmed the customer paid."
        : linkCreated
          ? "Waiting for the customer to pay the link."
          : "—",
      state: linkPaid ? "done" : linkCreated ? "current" : "pending",
      tone: "success",
    },
    {
      key: "recovered",
      title: "Recovery complete",
      detail: recovered
        ? "Outcome recorded, payment marked recovered, event closed."
        : eventStatus === "abandoned"
          ? "The automated campaign was exhausted without recovery."
          : "Not yet.",
      state: recovered ? "done" : eventStatus === "abandoned" ? "skipped" : "pending",
      tone: "success",
    },
  ];
}

export default function RecoveryDetail() {
  const { id } = useParams();
  const eventId = Number(id);
  const valid = Number.isInteger(eventId) && eventId > 0;

  const rzp = useRecoveryRazorpay(valid ? eventId : null);
  const event = useRecoveryEvent(valid ? eventId : null);
  const payment = usePayment(event.data?.payment_id ?? null);
  const subscription = useSubscription(payment.data?.subscription_id ?? null);
  const customer = useCustomer(subscription.data?.customer_id ?? null);
  const agentRun = useEventAgentRun(valid ? eventId : null);
  const agentEvents = useAgentEventList();

  const ctx = agentRun.data?.ctx ?? null;
  const merged = useMergedScores(
    valid ? eventId : 0,
    ctx?.quantitative_scores,
    !agentRun.isLoading,
  );

  const notFound = rzp.isError && rzp.error instanceof ApiError && rzp.error.status === 404;

  if (!valid || notFound) {
    return (
      <Page>
        <BackLink />
        <Card className="mt-4">
          <CardBody>
            <ErrorState error={new ApiError(404, `Recovery event ${id} doesn't exist.`)} />
          </CardBody>
        </Card>
      </Page>
    );
  }

  const r = rzp.data;
  const verified = eventId === VERIFIED_RECOVERY_EVENT_ID;
  const primaryIv = r?.interventions.find((i) => i.payment_link_created) ?? r?.interventions[0];
  const linkCreated = !!primaryIv?.payment_link_created;
  const linkPaid = !!primaryIv?.payment_link_paid;
  const recovered = !!r?.payment_recovered;

  const eventAgentEventItems =
    agentEvents.data?.filter((e) => e.recovery_event_id === eventId) ?? [];

  const stages = r
    ? buildStages({
        hasAgentRun: agentRun.found,
        chosenAction: ctx?.chosen_action ?? null,
        interventionExecuted: (r.interventions.length ?? 0) > 0,
        linkCreated,
        linkPaid,
        recovered,
        eventStatus: r.status,
        amountLabel: rupeesFromPaise(r.amount_paise),
        isControl: r.is_control,
      })
    : [];

  return (
    <Page>
      <BackLink />

      {/* header */}
      <div className="mb-7 mt-4 flex flex-col gap-3 border-b border-line/[.07] pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="label-caps mb-1.5">
            Recovery event #{eventId}
            {verified ? " · Razorpay Test Mode verified" : ""}
          </div>
          {rzp.isLoading ? (
            <Skeleton className="h-9 w-48" />
          ) : r ? (
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-3xl font-semibold tracking-tight text-ink tnum">
                {rupeesFromPaise(r.amount_paise, true)}
              </h1>
              <Badge
                tone={recovered ? "success" : r.status === "open" ? "warning" : "neutral"}
                dot
              >
                {recovered ? "Recovered" : r.status === "open" ? "In recovery" : "Not recovered"}
              </Badge>
              {r.is_control ? <Badge tone="neutral">Control</Badge> : null}
              {verified ? (
                <Badge tone="success">
                  <ShieldCheck size={11} /> Verified
                </Badge>
              ) : null}
            </div>
          ) : null}
          {payment.data ? (
            <p className="mt-2 text-[13px] text-ink-muted">
              {failureLabel(payment.data.failure_reason)} ·{" "}
              {payment.data.failed_at
                ? `failed ${relTime(payment.data.failed_at)}`
                : "—"}
              {customer.data?.email ? ` · ${customer.data.email}` : ""}
            </p>
          ) : null}
        </div>
        {event.data ? (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-right text-2xs">
            <dt className="text-ink-faint">Opened</dt>
            <dd className="font-mono text-ink-muted tnum">
              {dateTime(event.data.created_at)}
            </dd>
            {event.data.closed_at ? (
              <>
                <dt className="text-ink-faint">Closed</dt>
                <dd className="font-mono text-ink-muted tnum">
                  {dateTime(event.data.closed_at)}
                </dd>
              </>
            ) : null}
          </dl>
        ) : null}
      </div>

      {rzp.isError && !notFound ? (
        <Card>
          <CardBody>
            <ErrorState error={rzp.error} onRetry={() => rzp.refetch()} />
          </CardBody>
        </Card>
      ) : (
        <div className="grid gap-3 lg:grid-cols-[320px_minmax(0,1fr)]">
          {/* left: journey + evidence */}
          <div className="space-y-3">
            <Card>
              <CardHeader eyebrow="Lifecycle" title="Recovery journey" />
              <CardBody>
                {rzp.isLoading ? <SkeletonText lines={8} /> : <RecoveryJourney stages={stages} />}
              </CardBody>
            </Card>

            <Card>
              <CardHeader eyebrow="Trust" title="Evidence trail" />
              <CardBody>
                {agentEvents.isLoading ? (
                  <SkeletonText lines={4} />
                ) : (
                  <EvidenceTrail agentEvents={eventAgentEventItems} razorpay={r} />
                )}
              </CardBody>
            </Card>
          </div>

          {/* right: agent + intelligence + razorpay + message */}
          <div className="space-y-3">
            {agentRun.isLoading ? (
              <Card>
                <CardBody>
                  <SkeletonText lines={6} />
                </CardBody>
              </Card>
            ) : (
              <AgentRunPanel recoveryEventId={eventId} persisted={agentRun.data} />
            )}

            <Card>
              <CardHeader
                eyebrow="Quantitative intelligence"
                title="Intervention opportunity"
              />
              <CardBody>
                {merged.scores.length === 0 && !ctx ? (
                  <p className="text-[13px] text-ink-muted">
                    No untried actions to score — every action has already been attempted
                    on this event.
                  </p>
                ) : (
                  <InterventionIntelligence
                    scores={merged.scores}
                    chosenAction={ctx?.chosen_action ?? null}
                    source={merged.source}
                  />
                )}
              </CardBody>
            </Card>

            {r ? (
              <RazorpayCard
                interventions={r.interventions}
                config={r.razorpay_config}
                amountPaise={r.amount_paise}
              />
            ) : null}

            {ctx?.tool_trace.some((t) => t.tool === "execute_recovery_action") ||
            ctx?.chosen_action ? (
              <CustomerMessageCard
                message={
                  (ctx?.tool_trace.find((t) => t.tool === "execute_recovery_action")
                    ?.arguments?.customer_message as string | undefined) ?? null
                }
                audioUrl={ctx?.voice?.audio_url}
                voiceReason={ctx?.voice?.voice_reason}
                voiceEngine={ctx?.voice?.voice_engine}
              />
            ) : null}
          </div>
        </div>
      )}
    </Page>
  );
}

function BackLink() {
  return (
    <Link
      to="/recoveries"
      className="inline-flex items-center gap-1.5 text-[13px] text-ink-muted transition-colors hover:text-ink"
    >
      <ArrowLeft size={14} />
      All recoveries
    </Link>
  );
}
