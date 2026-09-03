import { Link } from "react-router-dom";
import { ArrowRight, BadgeCheck, Bot } from "lucide-react";
import { useAgentRunEvents } from "@/lib/queries";
import { VERIFIED_RECOVERY_EVENT_ID } from "@/lib/config";
import { Card, CardBody, CardHeader } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Skeleton } from "../ui/Skeleton";
import { ErrorState } from "../ui/States";
import { actionLabel, pct } from "@/lib/format";
import { cn } from "@/lib/cn";

export function FeaturedRecoveries() {
  const { runs: allRuns, isLoading, isError, error, refetch } = useAgentRunEvents();
  // Pin the Razorpay-verified recovery to the top of the list.
  const runs = [...allRuns].sort((a, b) => {
    const av = a.recovery_event_id === VERIFIED_RECOVERY_EVENT_ID ? 1 : 0;
    const bv = b.recovery_event_id === VERIFIED_RECOVERY_EVENT_ID ? 1 : 0;
    return bv - av || b.recovery_event_id - a.recovery_event_id;
  });

  return (
    <Card>
      <CardHeader
        eyebrow="Recovery intelligence"
        title="Agent decisions on record"
        action={
          <Link
            to="/recoveries"
            className="inline-flex items-center gap-1 text-2xs font-medium text-ink-muted hover:text-ink"
          >
            All recoveries <ArrowRight size={12} />
          </Link>
        }
      />
      <CardBody className="space-y-2">
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : isError ? (
          <ErrorState error={error} onRetry={() => refetch()} compact />
        ) : runs.length === 0 ? (
          <p className="py-6 text-center text-[13px] text-ink-muted">
            No agent runs recorded yet.
          </p>
        ) : (
          runs.slice(0, 6).map((run) => {
            const verified = run.recovery_event_id === VERIFIED_RECOVERY_EVENT_ID;
            const action = run.decision?.startsWith("execute:")
              ? run.decision.slice("execute:".length)
              : null;
            return (
              <Link
                key={run.id}
                to={`/recoveries/${run.recovery_event_id}`}
                className="group flex items-center gap-3 rounded-control border border-line/[.07] bg-surface-2/60 px-3.5 py-3 transition-colors hover:border-line/[.14] hover:bg-surface-2"
              >
                <span
                  className={cn(
                    "grid size-8 shrink-0 place-items-center rounded-full ring-1 ring-inset",
                    verified
                      ? "bg-success/10 text-success ring-success/25"
                      : "bg-accent/10 text-accent ring-accent/25",
                  )}
                >
                  {verified ? <BadgeCheck size={15} /> : <Bot size={15} />}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-semibold text-ink">
                      Recovery #{run.recovery_event_id}
                    </span>
                    {verified ? (
                      <Badge tone="success">Verified · Test Mode</Badge>
                    ) : null}
                  </div>
                  <p className="mt-0.5 truncate text-2xs text-ink-muted">
                    {action
                      ? `Agent executed ${actionLabel(action)}`
                      : run.decision?.replace(/[:_]/g, " ") ?? "—"}
                    {run.confidence != null
                      ? ` · model confidence ${pct(run.confidence)}`
                      : ""}
                  </p>
                </div>
                <ArrowRight
                  size={15}
                  className="shrink-0 text-ink-faint transition-transform group-hover:translate-x-0.5"
                />
              </Link>
            );
          })
        )}
      </CardBody>
    </Card>
  );
}
