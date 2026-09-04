import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ScrollText, ShieldCheck } from "lucide-react";
import { Page, PageHeader } from "@/components/layout/Shell";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState, ErrorState } from "@/components/ui/States";
import {
  useAuditLogList,
  useEventAgentRun,
  useRecoveryRazorpay,
  useAgentEventList,
} from "@/lib/queries";
import { EvidenceTrail } from "@/components/recovery/EvidenceTrail";
import { VERIFIED_RECOVERY_EVENT_ID } from "@/lib/config";
import { dateTime, num, titleCase } from "@/lib/format";
import { cn } from "@/lib/cn";

const PAGE = 40;

export default function Audit() {
  const logs = useAuditLogList();
  const [q, setQ] = useState("");
  const [actor, setActor] = useState("all");
  const [page, setPage] = useState(0);

  const verifiedRun = useEventAgentRun(VERIFIED_RECOVERY_EVENT_ID);
  const verifiedRzp = useRecoveryRazorpay(VERIFIED_RECOVERY_EVENT_ID);
  const agentEvents = useAgentEventList();

  const actors = useMemo(
    () => [...new Set(logs.data?.map((l) => l.actor) ?? [])].sort(),
    [logs.data],
  );

  const filtered = useMemo(() => {
    const rows = [...(logs.data ?? [])].reverse();
    const query = q.trim().toLowerCase();
    return rows.filter((r) => {
      if (actor !== "all" && r.actor !== actor) return false;
      if (query && !r.action.toLowerCase().includes(query)) return false;
      return true;
    });
  }, [logs.data, q, actor]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE));
  const clamped = Math.min(page, pageCount - 1);
  const rows = filtered.slice(clamped * PAGE, clamped * PAGE + PAGE);

  const verifiedEvents =
    agentEvents.data?.filter((e) => e.recovery_event_id === VERIFIED_RECOVERY_EVENT_ID) ?? [];

  return (
    <Page>
      <PageHeader
        eyebrow="Trust"
        title="Audit trail"
        description="Every decision, execution and recovery is a persisted record. No secrets, no raw payloads, no model chain-of-thought are ever stored."
      />

      {/* verified recovery evidence spotlight — the one story that matters,
          framed apart from the raw system log beneath it */}
      <Card className="mb-8 border-success/25 bg-gradient-to-br from-success/[.06] via-transparent to-transparent">
        <CardHeader
          eyebrow="The one recovery proven end to end"
          title={
            <span className="inline-flex items-center gap-2">
              <ShieldCheck size={15} className="text-success" />
              Recovery #{VERIFIED_RECOVERY_EVENT_ID} — Razorpay Test Mode, end to end
            </span>
          }
          action={
            <Link
              to={`/recoveries/${VERIFIED_RECOVERY_EVENT_ID}`}
              className="inline-flex items-center gap-1 text-2xs font-medium text-ink-muted hover:text-ink"
            >
              Open case file <ArrowRight size={12} />
            </Link>
          }
        />
        <CardBody>
          {verifiedRun.isLoading || verifiedRzp.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <EvidenceTrail agentEvents={verifiedEvents} razorpay={verifiedRzp.data} />
          )}
        </CardBody>
      </Card>

      {/* system-wide activity */}
      <h2 className="mb-3 flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.08em] text-ink-muted">
        <span className="h-3.5 w-0.5 rounded-full bg-accent/60" aria-hidden />
        Then everything else — the full system log
      </h2>
      <Card className="overflow-hidden">
        <CardHeader
          eyebrow="System"
          title="Activity log"
          action={
            <div className="flex items-center gap-2">
              <input
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setPage(0);
                }}
                placeholder="Filter actions…"
                aria-label="Filter audit actions"
                className="h-8 w-40 rounded-control bg-surface-2 px-2.5 text-[13px] text-ink outline-none ring-1 ring-inset ring-line/[.1] placeholder:text-ink-faint focus:ring-accent"
              />
              <select
                value={actor}
                onChange={(e) => {
                  setActor(e.target.value);
                  setPage(0);
                }}
                aria-label="Filter by actor"
                className="h-8 rounded-control bg-surface-2 px-2 text-[13px] text-ink outline-none ring-1 ring-inset ring-line/[.1] focus:ring-accent"
              >
                <option value="all">Any actor</option>
                {actors.map((a) => (
                  <option key={a} value={a}>
                    {titleCase(a)}
                  </option>
                ))}
              </select>
            </div>
          }
        />
        <CardBody className="p-0">
          {logs.isLoading ? (
            <div className="space-y-2 p-5">
              {Array.from({ length: 12 }).map((_, i) => (
                <Skeleton key={i} className="h-7 w-full" />
              ))}
            </div>
          ) : logs.isError ? (
            <div className="p-5">
              <ErrorState error={logs.error} onRetry={() => logs.refetch()} />
            </div>
          ) : rows.length === 0 ? (
            <EmptyState
              icon={<ScrollText size={18} />}
              title="No matching entries"
              description="Adjust the filter or actor."
            />
          ) : (
            <ul className="divide-y divide-line/[.05]">
              {rows.map((row) => (
                <li
                  key={row.id}
                  className="grid grid-cols-[110px_1fr_auto] items-center gap-3 px-5 py-2.5 text-[13px]"
                >
                  <Badge
                    tone={
                      row.actor.includes("agent")
                        ? "accent"
                        : row.actor.includes("webhook")
                          ? "success"
                          : "neutral"
                    }
                  >
                    {titleCase(row.actor)}
                  </Badge>
                  <span
                    className={cn(
                      "min-w-0 truncate",
                      row.actor.includes("webhook") || row.actor.includes("execution")
                        ? "font-medium text-ink"
                        : "text-ink-muted",
                    )}
                  >
                    {row.action.replace(/_/g, " ")}
                  </span>
                  <span className="shrink-0 font-mono text-2xs text-ink-faint tnum">
                    {dateTime(row.created_at)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>

      {!logs.isLoading && filtered.length > PAGE ? (
        <div className="mt-4 flex items-center justify-between text-[13px] text-ink-muted">
          <span className="tnum">
            {clamped * PAGE + 1}–{Math.min((clamped + 1) * PAGE, filtered.length)} of{" "}
            {num(filtered.length)}
          </span>
          <div className="flex items-center gap-2 font-mono text-2xs">
            <button
              className="link-quiet disabled:opacity-40"
              disabled={clamped === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              ← Prev
            </button>
            <span className="text-ink-faint">
              {clamped + 1} / {pageCount}
            </span>
            <button
              className="link-quiet disabled:opacity-40"
              disabled={clamped >= pageCount - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              Next →
            </button>
          </div>
        </div>
      ) : null}
    </Page>
  );
}
