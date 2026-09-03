import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export type Tone = "neutral" | "accent" | "success" | "warning" | "danger";

const TONES: Record<Tone, string> = {
  neutral: "bg-surface-3 text-ink-muted ring-1 ring-inset ring-line/[.1]",
  accent: "bg-accent/10 text-accent ring-1 ring-inset ring-accent/25",
  success: "bg-success/10 text-success ring-1 ring-inset ring-success/25",
  warning: "bg-warning/10 text-warning ring-1 ring-inset ring-warning/25",
  danger: "bg-danger/10 text-danger ring-1 ring-inset ring-danger/25",
};

export function Badge({
  children,
  tone = "neutral",
  dot = false,
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-chip px-2 py-0.5 text-2xs font-semibold",
        TONES[tone],
        className,
      )}
    >
      {dot ? (
        <span
          className={cn(
            "size-1.5 rounded-full",
            tone === "neutral" ? "bg-ink-faint" : "bg-current",
          )}
        />
      ) : null}
      {children}
    </span>
  );
}

/** Maps a recovery-event / payment status to a tone + label. */
export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { tone: Tone; label: string }> = {
    open: { tone: "warning", label: "In recovery" },
    closed: { tone: "success", label: "Recovered" },
    abandoned: { tone: "neutral", label: "Not recovered" },
    success: { tone: "success", label: "Paid" },
    failed: { tone: "danger", label: "Failed" },
    paid: { tone: "success", label: "Paid" },
    created: { tone: "accent", label: "Awaiting payment" },
    expired: { tone: "neutral", label: "Expired" },
    cancelled: { tone: "neutral", label: "Cancelled" },
    partially_paid: { tone: "warning", label: "Partially paid" },
  };
  const m = map[status] ?? { tone: "neutral" as Tone, label: status };
  return (
    <Badge tone={m.tone} dot>
      {m.label}
    </Badge>
  );
}
