import { useEffect, useState } from "react";
import {
  Ban,
  Check,
  ChevronDown,
  CircleDot,
  Eye,
  Flag,
  Gauge,
  ShieldAlert,
  Sparkles,
  Square,
  Zap,
} from "lucide-react";
import type { ActionIncrementality, ToolTraceEntry } from "@/lib/types";
import { phaseFor, type Phase } from "./toolMap";
import { IncrementalityCompare } from "./IncrementalityCompare";
import { actionLabel, ms } from "@/lib/format";
import { cn } from "@/lib/cn";

const INCREMENTALITY_TOOL = "get_historical_incrementality_for_action";

const KIND_ICON: Record<Phase["kind"], typeof Eye> = {
  observe: Eye,
  intelligence: Gauge,
  decide: Sparkles,
  execute: Zap,
  verify: Check,
  stop: Square,
  escalate: Flag,
};
const KIND_TONE: Record<Phase["kind"], string> = {
  observe: "text-ink-muted ring-line/[.12] bg-surface-3",
  intelligence: "text-accent ring-accent/30 bg-accent/10",
  decide: "text-accent ring-accent/30 bg-accent/10",
  execute: "text-accent ring-accent/30 bg-accent/10",
  verify: "text-success ring-success/30 bg-success/10",
  stop: "text-ink-muted ring-line/[.12] bg-surface-3",
  escalate: "text-warning ring-warning/30 bg-warning/10",
};

function TechnicalDetails({ entry }: { entry: ToolTraceEntry }) {
  const [open, setOpen] = useState(false);
  const hasArgs = entry.arguments && Object.keys(entry.arguments).length > 0;
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1 text-2xs font-medium text-ink-faint transition-colors hover:text-ink-muted"
        aria-expanded={open}
      >
        <ChevronDown size={12} className={cn("transition-transform", open && "rotate-180")} />
        Technical detail
      </button>
      {open ? (
        <div className="mt-2 space-y-2 rounded-[7px] border border-line/[.08] bg-bg/60 p-3 font-mono text-[11px] leading-relaxed text-ink-muted">
          <div>
            <span className="text-ink-faint">tool</span> {entry.tool}
          </div>
          {entry.result_summary ? (
            <div>
              <span className="text-ink-faint">result</span> {entry.result_summary}
            </div>
          ) : null}
          {entry.guardrail_code ? (
            <div className="text-warning">
              <span className="text-ink-faint">guardrail</span> {entry.guardrail_code}
            </div>
          ) : null}
          {hasArgs ? (
            <pre className="whitespace-pre-wrap break-words text-ink-muted">
              {JSON.stringify(entry.arguments, null, 2)}
            </pre>
          ) : null}
          <div className="text-ink-faint">
            {entry.latency_ms != null ? `model latency ${ms(entry.latency_ms)} · ` : ""}
            {entry.prompt_tokens != null
              ? `${entry.prompt_tokens + (entry.output_tokens ?? 0)} tokens`
              : ""}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Row({
  entry,
  index,
  animate,
  incrementality,
}: {
  entry: ToolTraceEntry;
  index: number;
  animate: boolean;
  incrementality?: ActionIncrementality;
}) {
  const phase = phaseFor(entry.tool);
  const Icon = entry.ok
    ? KIND_ICON[phase.kind]
    : phase.kind === "execute"
      ? Ban
      : ShieldAlert;
  const tone = entry.ok ? KIND_TONE[phase.kind] : "text-danger ring-danger/30 bg-danger/10";
  const action =
    typeof entry.arguments?.action_type === "string"
      ? actionLabel(entry.arguments.action_type as string)
      : null;
  const labelTone = !entry.ok
    ? "text-danger"
    : phase.kind === "intelligence" || phase.kind === "execute" || phase.kind === "decide"
      ? "text-accent"
      : phase.kind === "verify"
        ? "text-success"
        : phase.kind === "escalate"
          ? "text-warning"
          : "text-ink-muted";

  return (
    <li
      className={cn("relative grid grid-cols-[auto_minmax(0,1fr)] gap-x-3.5", animate && "animate-fade-up")}
      style={animate ? { animationDelay: `${index * 70}ms` } : undefined}
    >
      <div className="flex flex-col items-center">
        <span
          className={cn(
            "grid size-7 shrink-0 place-items-center rounded-full ring-1 ring-inset",
            tone,
          )}
        >
          <Icon size={13} />
        </span>
        <span className="w-px flex-1 bg-line/[.1] last:hidden" aria-hidden />
      </div>

      <div className="min-w-0 pb-5">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-mono text-[11px] text-ink-faint tnum">
            {String(entry.turn).padStart(2, "0")}
          </span>
          <span className={cn("text-[11px] font-semibold uppercase tracking-[0.06em]", labelTone)}>
            {phase.label}
          </span>
          {action ? (
            <span className="rounded-chip bg-surface-3 px-1.5 py-0.5 text-2xs font-medium text-ink-muted">
              {action}
            </span>
          ) : null}
          {entry.ok ? (
            <Check size={12} className="text-success" />
          ) : (
            <span className="text-2xs font-medium text-danger">rejected</span>
          )}
        </div>
        <p className="mt-1 text-[13px] leading-snug text-ink">{phase.description}</p>
        {entry.guardrail_code && !entry.ok ? (
          <p className="mt-1 text-2xs text-warning">
            Guardrail: {entry.guardrail_code.replace(/_/g, " ")} — the agent must choose a
            different step.
          </p>
        ) : null}
        {entry.tool === INCREMENTALITY_TOOL && entry.ok && incrementality ? (
          <IncrementalityCompare data={incrementality} />
        ) : null}
        <TechnicalDetails entry={entry} />
      </div>
    </li>
  );
}

export function AgentTimeline({
  trace,
  animate = true,
  live = false,
  actionIncrementality,
}: {
  trace: ToolTraceEntry[];
  animate?: boolean;
  live?: boolean;
  actionIncrementality?: Record<string, ActionIncrementality> | null;
}) {
  const [revealed, setRevealed] = useState(live ? 0 : trace.length);

  useEffect(() => {
    if (!live) {
      setRevealed(trace.length);
      return;
    }
    setRevealed(0);
    let i = 0;
    const timer = window.setInterval(() => {
      i += 1;
      setRevealed(i);
      if (i >= trace.length) window.clearInterval(timer);
    }, 560);
    return () => window.clearInterval(timer);
  }, [trace, live]);

  const shown = trace.slice(0, revealed);

  if (trace.length === 0) {
    return (
      <div className="flex items-center gap-2 px-1 py-4 text-[13px] text-ink-muted">
        <CircleDot size={14} className="text-ink-faint" />
        No tool calls were recorded for this run.
      </div>
    );
  }

  return (
    <ol className="mt-1">
      {shown.map((entry, i) => (
        <Row
          key={`${entry.turn}-${entry.tool}-${i}`}
          entry={entry}
          index={i}
          animate={animate}
          incrementality={
            typeof entry.arguments?.action_type === "string"
              ? actionIncrementality?.[entry.arguments.action_type as string]
              : undefined
          }
        />
      ))}
      {live && revealed < trace.length ? (
        <li className="flex items-center gap-3 pl-[2px]">
          <span className="grid size-7 place-items-center rounded-full bg-accent/10 ring-1 ring-inset ring-accent/30">
            <span className="size-2 animate-pulse rounded-full bg-accent" />
          </span>
          <span className="text-[13px] text-ink-muted">Reasoning…</span>
        </li>
      ) : null}
    </ol>
  );
}
