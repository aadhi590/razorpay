import { useState } from "react";
import { Brain, Sparkles } from "lucide-react";
import type { QuantScore } from "@/lib/types";
import { RankedBars, type RankedItem } from "../ui/Bars";
import { actionLabel, pct, rupeesFromPaise } from "@/lib/format";
import { cn } from "@/lib/cn";

type View = "ml" | "uplift";

export function InterventionIntelligence({
  scores,
  chosenAction,
  source,
  className,
}: {
  scores: QuantScore[];
  chosenAction?: string | null;
  source?: string | null;
  className?: string;
}) {
  const hasUplift = scores.some((s) => s.uplift != null);
  const hasMl = scores.some((s) => s.recovery_probability != null);
  const [view, setView] = useState<View>(hasMl ? "ml" : "uplift");

  if (scores.length === 0) {
    return (
      <p className="px-1 py-4 text-[13px] text-ink-muted">
        No eligible actions were scored for this event.
      </p>
    );
  }

  const items: RankedItem[] =
    view === "ml"
      ? [...scores]
          .filter((s) => s.recovery_probability != null)
          .sort((a, b) => (b.expected_value_paise ?? 0) - (a.expected_value_paise ?? 0))
          .map((s) => ({
            label: actionLabel(s.action),
            value: s.recovery_probability ?? 0,
            display: pct(s.recovery_probability, 1),
            highlight: s.action === chosenAction,
            caption:
              s.expected_value_paise != null
                ? `Expected value ${rupeesFromPaise(Math.round(s.expected_value_paise))} · cost ${rupeesFromPaise(s.cost_paise, true)}`
                : undefined,
          }))
      : [...scores]
          .filter((s) => s.uplift != null)
          .sort(
            (a, b) =>
              (b.net_incremental_value_paise ?? 0) - (a.net_incremental_value_paise ?? 0),
          )
          .map((s) => ({
            label: actionLabel(s.action),
            value: Math.max(0, s.uplift ?? 0),
            display: `+${pct(s.uplift, 1)}`,
            highlight: s.action === chosenAction,
            caption:
              s.net_incremental_value_paise != null
                ? `Net incremental value ${rupeesFromPaise(Math.round(s.net_incremental_value_paise))}`
                : undefined,
          }));

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-control bg-surface-3 p-0.5 text-2xs font-semibold">
          {hasMl ? (
            <button
              onClick={() => setView("ml")}
              className={cn(
                "flex items-center gap-1.5 rounded-[7px] px-2.5 py-1.5 transition-colors",
                view === "ml" ? "bg-surface text-ink shadow-panel" : "text-ink-faint",
              )}
            >
              <Brain size={12} />
              Recovery probability
            </button>
          ) : null}
          {hasUplift ? (
            <button
              onClick={() => setView("uplift")}
              className={cn(
                "flex items-center gap-1.5 rounded-[7px] px-2.5 py-1.5 transition-colors",
                view === "uplift" ? "bg-surface text-ink shadow-panel" : "text-ink-faint",
              )}
            >
              <Sparkles size={12} />
              Causal uplift
            </button>
          ) : null}
        </div>
        {source ? (
          <span className="font-mono text-2xs text-ink-faint">source: {source}</span>
        ) : null}
      </div>

      <RankedBars items={items} />

      <p className="border-t border-line/[.07] pt-3 text-2xs leading-relaxed text-ink-faint">
        {view === "ml" ? (
          <>
            <span className="font-semibold text-ink-muted">ML</span> estimates the
            calibrated probability each action recovers the payment, and its expected
            value net of cost.
          </>
        ) : (
          <>
            <span className="font-semibold text-ink-muted">Uplift</span> estimates the{" "}
            <span className="italic">incremental</span> effect of each action versus no
            intervention — a customer likely to pay anyway is correctly deprioritised.
          </>
        )}{" "}
        The AI agent chooses the intervention; the models only inform it.
      </p>
    </div>
  );
}
