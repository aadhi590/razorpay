import type { ReactNode } from "react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/cn";

export function Stat({
  label,
  value,
  sub,
  delta,
  hero = false,
  tone = "neutral",
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  delta?: { value: string; positive: boolean } | null;
  hero?: boolean;
  tone?: "neutral" | "success" | "warning" | "accent" | "danger";
  className?: string;
}) {
  const toneText =
    tone === "success"
      ? "text-success"
      : tone === "warning"
        ? "text-warning"
        : tone === "danger"
          ? "text-danger"
          : tone === "accent"
            ? "text-accent"
            : "text-ink";
  return (
    <div className={cn("panel flex flex-col p-5 shadow-panel", className)}>
      <div className="label-caps">{label}</div>
      <div
        className={cn(
          "mt-2 font-semibold tracking-tight tnum",
          hero ? "text-[clamp(1.9rem,4vw,2.9rem)] leading-none" : "text-2xl",
          toneText,
        )}
      >
        {value}
      </div>
      <div className="mt-2 flex items-center gap-2 text-[13px]">
        {delta ? (
          <span
            className={cn(
              "inline-flex items-center gap-0.5 font-medium tnum",
              delta.positive ? "text-success" : "text-danger",
            )}
          >
            {delta.positive ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
            {delta.value}
          </span>
        ) : null}
        {sub ? <span className="text-ink-muted">{sub}</span> : null}
      </div>
    </div>
  );
}
