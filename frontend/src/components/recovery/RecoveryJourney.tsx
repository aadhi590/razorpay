import { Check, CircleAlert, CircleDashed, Minus } from "lucide-react";
import { cn } from "@/lib/cn";

export type StageState = "done" | "current" | "pending" | "skipped";

export interface Stage {
  key: string;
  title: string;
  detail?: string;
  state: StageState;
  tone?: "accent" | "success" | "warning";
}

const STATE_META: Record<
  StageState,
  { icon: typeof Check; ring: string; text: string; line: string }
> = {
  done: {
    icon: Check,
    ring: "bg-success/12 text-success ring-success/30",
    text: "text-ink",
    line: "bg-success/40",
  },
  current: {
    icon: CircleAlert,
    ring: "bg-accent/12 text-accent ring-accent/40 animate-pulse-ring",
    text: "text-ink",
    line: "bg-line/[.12]",
  },
  pending: {
    icon: CircleDashed,
    ring: "bg-surface-3 text-ink-faint ring-line/[.1]",
    text: "text-ink-faint",
    line: "bg-line/[.1]",
  },
  skipped: {
    icon: Minus,
    ring: "bg-surface-3 text-ink-faint ring-line/[.1]",
    text: "text-ink-faint",
    line: "bg-line/[.1]",
  },
};

export function RecoveryJourney({ stages }: { stages: Stage[] }) {
  return (
    <ol className="relative">
      {stages.map((stage, i) => {
        const meta = STATE_META[stage.state];
        // An accent-toned stage that has actually happened (the agent's
        // decision) renders in the accent — its defined "AI decision" meaning.
        const accentDone = stage.tone === "accent" && stage.state === "done";
        const ring = accentDone
          ? "bg-accent/12 text-accent ring-accent/35"
          : meta.ring;
        const Icon = stage.tone === "success" ? Check : meta.icon;
        const isLast = i === stages.length - 1;
        return (
          <li
            key={stage.key}
            className="grid animate-fade-up grid-cols-[auto_minmax(0,1fr)] gap-x-3.5"
            style={{ animationDelay: `${i * 45}ms` }}
          >
            <div className="flex flex-col items-center">
              <span
                className={cn(
                  "grid size-8 shrink-0 place-items-center rounded-full ring-1 ring-inset transition-colors",
                  ring,
                )}
              >
                <Icon size={14} strokeWidth={2.4} />
              </span>
              {!isLast ? (
                <span className={cn("w-px flex-1", meta.line)} aria-hidden />
              ) : null}
            </div>
            <div className={cn("min-w-0", isLast ? "pb-0" : "pb-6")}>
              <p
                className={cn(
                  "text-[13px] font-semibold leading-tight",
                  meta.text,
                  (stage.state === "current" || accentDone) && "text-accent",
                )}
              >
                {stage.title}
              </p>
              {stage.detail ? (
                <p
                  className={cn(
                    "mt-1 text-2xs leading-relaxed",
                    stage.state === "pending" || stage.state === "skipped"
                      ? "text-ink-faint"
                      : "text-ink-muted",
                  )}
                >
                  {stage.detail}
                </p>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
