import { useState } from "react";
import {
  AlertTriangle,
  Bot,
  Clock,
  Cpu,
  Flag,
  Play,
  RotateCw,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import type { AgentRunResult, ToolTraceEntry } from "@/lib/types";
import type { EventAgentRun } from "@/lib/queries";
import { useRunAgent } from "@/lib/queries";
import { AgentTimeline } from "./AgentTimeline";
import { stopReasonLabel } from "./toolMap";
import { Card, CardBody, CardHeader } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { ErrorState } from "../ui/States";
import { useToast } from "../ui/Toast";
import { actionLabel, ms, num } from "@/lib/format";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/cn";

interface Props {
  recoveryEventId: number;
  persisted?: EventAgentRun;
}

function DecisionSummary({
  decision,
  chosenAction,
  reasoning,
  status,
  stopReason,
  escalationType,
  turns,
  latencyMs,
  tokens,
  model,
}: {
  decision: string;
  chosenAction: string | null;
  reasoning: string;
  status: string;
  stopReason: string;
  escalationType?: string | null;
  turns: number;
  latencyMs: number;
  tokens?: number;
  model: string;
}) {
  const escalated = status === "escalated";
  const failedSafe = status === "failed_safe";
  return (
    <div
      className={cn(
        "rounded-control border p-4",
        escalated
          ? "border-warning/25 bg-warning/[.06]"
          : failedSafe
            ? "border-line/[.1] bg-surface-2"
            : "border-accent/25 bg-accent/[.05]",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "grid size-7 place-items-center rounded-full",
            escalated
              ? "bg-warning/15 text-warning"
              : failedSafe
                ? "bg-surface-3 text-ink-muted"
                : "bg-accent/15 text-accent",
          )}
        >
          {escalated ? <Flag size={14} /> : failedSafe ? <TriangleAlert size={14} /> : <Sparkles size={14} />}
        </span>
        <div>
          <p className="label-caps">Agent decision</p>
          <p className="text-[14px] font-semibold text-ink">
            {chosenAction
              ? `Execute ${actionLabel(chosenAction)}`
              : escalated
                ? `Escalate — ${escalationType?.replace(/_/g, " ") ?? "human review"}`
                : stopReasonLabel(stopReason)}
          </p>
        </div>
      </div>

      {reasoning ? (
        <p className="mt-3 border-l-2 border-line/[.12] pl-3 text-[13px] leading-relaxed text-ink-muted">
          {reasoning}
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-line/[.08] pt-3 text-2xs text-ink-faint">
        <span className="inline-flex items-center gap-1">
          <Bot size={11} />{" "}
          {chosenAction
            ? `executed ${actionLabel(chosenAction)}`
            : decision.replace(/[:_]/g, " ")}
        </span>
        <span className="inline-flex items-center gap-1">
          <Clock size={11} /> {turns} turn{turns === 1 ? "" : "s"} · {ms(latencyMs)}
        </span>
        {tokens ? (
          <span className="inline-flex items-center gap-1">
            <Cpu size={11} /> {num(tokens)} tokens
          </span>
        ) : null}
        <span className="font-mono">{model}</span>
      </div>
    </div>
  );
}

export function AgentRunPanel({ recoveryEventId, persisted }: Props) {
  const toast = useToast();
  const runMut = useRunAgent();
  const [liveResult, setLiveResult] = useState<AgentRunResult | null>(null);
  const [replayKey, setReplayKey] = useState(0);

  const ctx = persisted?.ctx ?? null;

  // The trace + summary currently displayed: a fresh live run takes precedence.
  const trace: ToolTraceEntry[] = liveResult?.tool_trace ?? ctx?.tool_trace ?? [];
  const actionIncrementality =
    liveResult?.action_incrementality ?? ctx?.action_incrementality ?? null;
  const actionLiftTrend =
    liveResult?.action_lift_trend ?? ctx?.action_lift_trend ?? null;

  // A rate-limited / provider-outage run stores the raw provider error as its
  // reasoning. When the agent had already chosen an action, its real rationale
  // is the `reason` it gave the execute tool — show that instead.
  const executeReason = (() => {
    const ex = [...trace].reverse().find((t) => t.tool === "execute_recovery_action");
    const r = ex?.arguments?.reason;
    return typeof r === "string" && r.trim() ? r.trim() : null;
  })();
  const cleanReason = (raw: string): string => {
    if (/http\s*4\d\d|quota exceeded|rate limit|https?:\/\//i.test(raw)) {
      return (
        executeReason ??
        "The agent selected this action, then reached the Gemini free-tier rate limit and stopped safely before the run could formally conclude."
      );
    }
    return raw;
  };

  const summary = liveResult
    ? {
        decision: liveResult.decision,
        chosenAction: liveResult.chosen_action,
        reasoning: cleanReason(liveResult.reasoning_summary),
        status: liveResult.status,
        stopReason: liveResult.stop_reason,
        escalationType: liveResult.escalation_type,
        turns: liveResult.turns_used,
        latencyMs: liveResult.latency_ms,
        tokens: liveResult.token_usage?.total_tokens,
        model: liveResult.model,
      }
    : ctx
      ? {
          decision: ctx.decision,
          chosenAction: ctx.chosen_action,
          reasoning: cleanReason(ctx.reasoning_summary),
          status: ctx.status,
          stopReason: ctx.stop_reason,
          escalationType: ctx.escalation_type,
          turns: ctx.turns_used,
          latencyMs: ctx.latency_ms,
          tokens: ctx.token_usage?.total_tokens,
          model: ctx.model,
        }
      : null;

  const isScripted = summary?.model?.startsWith("demo-") ?? false;
  const runLive = () => {
    setLiveResult(null);
    runMut.mutate(
      { id: recoveryEventId, dryRun: true },
      {
        onSuccess: (data) => {
          setLiveResult(data);
          setReplayKey((k) => k + 1);
          if (data.status === "failed_safe") {
            toast.push(
              data.stop_reason === "quota_or_api_failure"
                ? "Gemini is rate-limited right now — the agent degraded safely. Your data is untouched."
                : "The agent terminated safely. Nothing was changed.",
              "info",
            );
          } else {
            toast.push("Live agent run complete.", "success");
          }
        },
        onError: (e) => {
          toast.push(
            e instanceof ApiError ? e.detail : "The live run could not start.",
            "error",
          );
        },
      },
    );
  };

  return (
    <Card>
      <CardHeader
        eyebrow="Recovery intelligence"
        title="AI agent"
        action={
          <div className="flex items-center gap-2">
            {liveResult ? (
              <Badge tone="accent" dot>
                Live run
              </Badge>
            ) : persisted ? (
              <Badge tone="neutral" dot>
                Recorded run
              </Badge>
            ) : null}
          </div>
        }
      />
      <CardBody className="space-y-4">
        {!persisted && !liveResult && !runMut.isPending ? (
          <div className="rounded-control border border-line/[.08] bg-surface-2 p-4">
            <p className="text-[13px] text-ink-muted">
              No agent run is on record for this event. Run the autonomous agent to
              see it observe the context, score every action, and decide — one tool
              per turn.
            </p>
          </div>
        ) : null}

        {runMut.isPending ? (
          <div className="flex items-center gap-3 rounded-control border border-accent/25 bg-accent/[.05] p-4">
            <span className="grid size-8 place-items-center rounded-full bg-accent/15">
              <span className="size-2.5 animate-pulse rounded-full bg-accent" />
            </span>
            <div>
              <p className="flex items-center gap-1.5 text-[13px] font-semibold text-accent">
                AI recovery active
              </p>
              <p className="text-2xs text-ink-muted">
                Calling Gemini in real time — this can take up to a minute on the free
                tier.
              </p>
            </div>
          </div>
        ) : null}

        {runMut.isError && !liveResult ? (
          <ErrorState error={runMut.error} onRetry={runLive} compact />
        ) : null}

        {summary ? (
          <>
            <div key={replayKey}>
              <AgentTimeline
                trace={trace}
                live={!!liveResult}
                animate
                actionIncrementality={actionIncrementality}
                actionLiftTrend={actionLiftTrend}
              />
            </div>

            <DecisionSummary {...summary} />

            {summary.status === "failed_safe" ? (
              <div className="flex items-start gap-2 rounded-control border border-line/[.08] bg-surface-2 p-3 text-2xs leading-relaxed text-ink-muted">
                <AlertTriangle size={13} className="mt-0.5 shrink-0 text-warning" />
                <span>
                  This run ended in a <span className="font-medium text-ink">safe stop</span> (
                  {stopReasonLabel(summary.stopReason)}). Guardrails and the database are
                  unaffected — a created Payment Link is never treated as a recovery, and
                  no state is corrupted.
                </span>
              </div>
            ) : null}

            {isScripted ? (
              <p className="text-2xs text-ink-faint">
                Recorded with a deterministic provider (no Gemini calls). Use{" "}
                <span className="font-medium text-ink-muted">Run live</span> for a real
                Gemini run.
              </p>
            ) : null}
          </>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 border-t border-line/[.07] pt-4">
          <Button variant="primary" size="sm" onClick={runLive} loading={runMut.isPending}>
            {liveResult || persisted ? <RotateCw size={13} /> : <Play size={13} />}
            {liveResult || persisted ? "Run live again" : "Run recovery agent"}
          </Button>
          <span className="text-2xs text-ink-faint">
            Dry run · calls Gemini · subject to free-tier rate limits · no external
            action is executed
          </span>
        </div>
      </CardBody>
    </Card>
  );
}
