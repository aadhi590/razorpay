import { useState } from "react";
import { Link } from "react-router-dom";
import { Scissors } from "lucide-react";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Stat } from "@/components/ui/Stat";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/States";
import { usePortfolioAllocation } from "@/lib/queries";
import type { PortfolioAllocationEvent } from "@/lib/types";
import { actionLabel, num, pct, rupeesCompactFromPaise, rupeesFromPaise } from "@/lib/format";
import { cn } from "@/lib/cn";

const CAPACITY_OPTIONS = [1, 2, 3, 5, 10];

/**
 * The batch-scarcity view: given a limited number of interventions the ops team
 * can run right now, which currently-open recovery events get acted on, which
 * get skipped, and — the payoff stat — how much expected value the capacity
 * limit costs versus acting on everything worth acting on.
 *
 * Reuses the existing Stat / Badge / table conventions and only the existing
 * state colours (accent = acted, neutral/faint = skipped, warning on the
 * cutoff). No new visual pattern, no new route.
 */
export function PortfolioAllocation() {
  const [capacity, setCapacity] = useState(2);
  const q = usePortfolioAllocation(capacity);

  return (
    <Card>
      <CardHeader
        eyebrow="Scarcity allocator"
        title="Allocation under capacity"
        action={
          <label className="flex items-center gap-2 text-2xs text-ink-faint">
            Capacity
            <select
              value={capacity}
              onChange={(e) => setCapacity(Number(e.target.value))}
              aria-label="Interventions available this batch"
              className="h-8 rounded-control bg-surface-2 px-2 text-[13px] text-ink outline-none ring-1 ring-inset ring-line/[.1] focus:ring-accent"
            >
              {CAPACITY_OPTIONS.map((c) => (
                <option key={c} value={c}>
                  {c} intervention{c === 1 ? "" : "s"}
                </option>
              ))}
            </select>
          </label>
        }
      />
      <CardBody className="space-y-5">
        {q.isLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} compact />
        ) : !q.data ? null : !q.data.computable ? (
          <p className="text-[13px] text-ink-muted">
            {q.data.reason ??
              "No open, eligible recovery events to allocate right now."}
          </p>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat
                label="Expected value the limit costs"
                value={rupeesCompactFromPaise(
                  q.data.expected_value_forgone_to_capacity_paise,
                )}
                sub={`capacity ${q.data.capacity_used} of ${num(q.data.events_ranked)} eligible`}
                tone={
                  q.data.expected_value_forgone_to_capacity_paise > 0
                    ? "warning"
                    : "success"
                }
                hero
              />
              <Stat
                label="Captured by acting"
                value={rupeesCompactFromPaise(
                  q.data.expected_value_captured_paise,
                )}
                sub={`top ${num(q.data.act.length)} by expected value`}
                tone="accent"
              />
              <Stat
                label="If capacity were unlimited"
                value={rupeesCompactFromPaise(
                  q.data.expected_value_if_unlimited_paise,
                )}
                sub="act on every positive-value event"
              />
            </div>

            <div className="overflow-x-auto">
              <div className="min-w-[560px]">
                <div className="grid grid-cols-[44px_1fr_110px_150px_120px] gap-3 border-b border-line/[.07] px-1 pb-2 text-2xs font-semibold uppercase tracking-[0.06em] text-ink-faint">
                  <span>Rank</span>
                  <span>Event</span>
                  <span className="text-right">Amount</span>
                  <span>Best action</span>
                  <span className="text-right">Expected value</span>
                </div>

                <ul>
                  {q.data.act.map((e) => (
                    <AllocationRow key={e.recovery_event_id} e={e} acted />
                  ))}

                  {q.data.skip.length > 0 ? (
                    <li
                      className="my-1 flex items-center gap-2 rounded-control bg-warning/[.08] px-2.5 py-1.5 text-2xs font-semibold text-warning"
                      aria-hidden
                    >
                      <Scissors size={12} />
                      Capacity cutoff — {q.data.capacity} intervention
                      {q.data.capacity === 1 ? "" : "s"} available
                    </li>
                  ) : null}

                  {q.data.skip.map((e) => (
                    <AllocationRow key={e.recovery_event_id} e={e} />
                  ))}
                </ul>
              </div>
            </div>

            <p className="border-t border-line/[.07] pt-3 text-2xs leading-relaxed text-ink-faint">
              {q.data.note}
            </p>
          </>
        )}
      </CardBody>
    </Card>
  );
}

function AllocationRow({
  e,
  acted = false,
}: {
  e: PortfolioAllocationEvent;
  acted?: boolean;
}) {
  return (
    <li
      className={cn(
        "grid grid-cols-[44px_1fr_110px_150px_120px] items-start gap-3 border-t border-line/[.05] px-1 py-2.5 text-[13px] first:border-t-0",
        !acted && "opacity-80",
      )}
    >
      <span className="pt-0.5 font-mono text-2xs text-ink-faint tnum">
        {e.rank != null ? `#${e.rank}` : "—"}
      </span>
      <span className="min-w-0">
        <Link
          to={`/recoveries/${e.recovery_event_id}`}
          className="font-mono text-[12px] text-ink-muted tnum hover:text-accent"
        >
          #{e.recovery_event_id}
        </Link>
        <span className="ml-2">
          <Badge tone={acted ? "accent" : "neutral"} dot>
            {acted ? "Act" : "Skip"}
          </Badge>
        </span>
        <p className="mt-1 text-2xs leading-relaxed text-ink-faint">{e.reason}</p>
      </span>
      <span className="pt-0.5 text-right font-medium text-ink tnum">
        {rupeesFromPaise(e.amount_paise)}
      </span>
      <span className="pt-0.5 text-2xs text-ink-muted">
        {e.best_action ? (
          <>
            {actionLabel(e.best_action)}
            {e.recovery_probability != null ? (
              <span className="block text-ink-faint">
                p(recover) {pct(e.recovery_probability)}
              </span>
            ) : null}
          </>
        ) : (
          "—"
        )}
      </span>
      <span
        className={cn(
          "pt-0.5 text-right font-mono tnum",
          acted ? "font-semibold text-accent" : "text-ink-muted",
        )}
      >
        {e.expected_value_paise != null
          ? rupeesCompactFromPaise(e.expected_value_paise)
          : "—"}
      </span>
    </li>
  );
}
