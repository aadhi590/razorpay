import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Bot, Search, ShieldCheck } from "lucide-react";
import { Page, PageHeader } from "@/components/layout/Shell";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState, ErrorState } from "@/components/ui/States";
import { useRecoveryIndex, type RecoveryRow } from "@/lib/queries";
import { PAGE_SIZE, VERIFIED_RECOVERY_EVENT_ID } from "@/lib/config";
import { actionLabel, num, rupeesFromPaise, relTime } from "@/lib/format";
import { cn } from "@/lib/cn";

type StatusFilter = "all" | "recovered" | "in_recovery" | "not_recovered";

const STATUS_TABS: { key: StatusFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "recovered", label: "Recovered" },
  { key: "in_recovery", label: "In recovery" },
  { key: "not_recovered", label: "Not recovered" },
];

function matchStatus(row: RecoveryRow, f: StatusFilter): boolean {
  if (f === "all") return true;
  if (f === "recovered") return row.recovered;
  if (f === "in_recovery") return row.status === "open" && !row.recovered;
  return !row.recovered && row.status !== "open";
}

export default function Recoveries() {
  const { rows, isLoading, isError, error } = useRecoveryIndex();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [action, setAction] = useState<string>("all");
  const [agentOnly, setAgentOnly] = useState(false);
  const [page, setPage] = useState(0);

  const actionOptions = useMemo(
    () =>
      [
        ...new Set(
          rows.flatMap((r) => [...r.actions, ...(r.agentAction ? [r.agentAction] : [])]),
        ),
      ].sort(),
    [rows],
  );

  const filtered = useMemo(() => {
    const query = q.trim();
    return rows.filter((r) => {
      if (query && !String(r.id).includes(query.replace(/^#/, ""))) return false;
      if (!matchStatus(r, status)) return false;
      if (action !== "all" && !r.actions.includes(action) && r.agentAction !== action)
        return false;
      if (agentOnly && !r.hasAgentRun) return false;
      return true;
    });
  }, [rows, q, status, action, agentOnly]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const pageRows = filtered.slice(
    clampedPage * PAGE_SIZE,
    clampedPage * PAGE_SIZE + PAGE_SIZE,
  );

  const resetPage = () => setPage(0);

  return (
    <Page>
      <PageHeader
        eyebrow="Explorer"
        title="Recoveries"
        description={
          isLoading
            ? "Loading the recovery portfolio…"
            : `${num(filtered.length)} recovery event${filtered.length === 1 ? "" : "s"}${
                filtered.length !== rows.length ? ` of ${num(rows.length)}` : ""
              }`
        }
      />

      {isError ? (
        <Card className="p-2">
          <ErrorState error={error} />
        </Card>
      ) : (
        <>
          {/* controls — a wide "working list" console, distinct from a stat summary */}
          <div className="mb-4 flex flex-col gap-3 rounded-card border border-line/[.08] bg-surface-2/50 p-2.5 shadow-panel lg:flex-row lg:items-center lg:justify-between">
            <div className="inline-flex rounded-control bg-surface-3 p-0.5 text-2xs font-semibold">
              {STATUS_TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => {
                    setStatus(t.key);
                    resetPage();
                  }}
                  className={cn(
                    "rounded-[7px] px-3 py-1.5 transition-colors",
                    status === t.key ? "bg-surface text-ink shadow-panel" : "text-ink-faint",
                  )}
                >
                  {t.label}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2 lg:flex-1 lg:justify-end">
              <label className="relative grow sm:grow-0">
                <Search
                  size={14}
                  className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint"
                />
                <input
                  value={q}
                  onChange={(e) => {
                    setQ(e.target.value);
                    resetPage();
                  }}
                  inputMode="numeric"
                  placeholder="Search event ID…"
                  aria-label="Search by recovery event ID"
                  className="h-8 w-full rounded-control bg-surface pl-8 pr-2 text-[13px] text-ink outline-none ring-1 ring-inset ring-line/[.1] placeholder:text-ink-faint focus:ring-accent sm:w-52"
                />
              </label>
              <select
                value={action}
                onChange={(e) => {
                  setAction(e.target.value);
                  resetPage();
                }}
                aria-label="Filter by action"
                className="h-8 rounded-control bg-surface-2 px-2 text-[13px] text-ink outline-none ring-1 ring-inset ring-line/[.1] focus:ring-accent"
              >
                <option value="all">Any action</option>
                {actionOptions.map((a) => (
                  <option key={a} value={a}>
                    {actionLabel(a)}
                  </option>
                ))}
              </select>
              <button
                onClick={() => {
                  setAgentOnly((v) => !v);
                  resetPage();
                }}
                className={cn(
                  "inline-flex h-8 items-center gap-1.5 rounded-control px-2.5 text-2xs font-semibold ring-1 ring-inset transition-colors",
                  agentOnly
                    ? "bg-accent/12 text-accent ring-accent/25"
                    : "bg-surface-2 text-ink-faint ring-line/[.1] hover:text-ink-muted",
                )}
              >
                <Bot size={13} />
                Agent runs
              </button>
            </div>
          </div>

          <Card className="overflow-hidden">
            {/* header row */}
            <div className="hidden grid-cols-[80px_1fr_120px_130px_110px_90px] gap-3 border-b border-line/[.07] px-4 py-2.5 text-2xs font-semibold uppercase tracking-[0.06em] text-ink-faint sm:grid">
              <span>Event</span>
              <span>Recovery</span>
              <span>Amount</span>
              <span>Action</span>
              <span>Status</span>
              <span className="text-right">Opened</span>
            </div>

            {isLoading ? (
              <div className="space-y-px">
                {Array.from({ length: 10 }).map((_, i) => (
                  <div key={i} className="px-4 py-3">
                    <Skeleton className="h-8 w-full" />
                  </div>
                ))}
              </div>
            ) : pageRows.length === 0 ? (
              <EmptyState
                title="No matching recoveries"
                description="Adjust the filters or search a different event ID."
              />
            ) : (
              <ul
                key={`${status}|${action}|${agentOnly}|${clampedPage}`}
                className="animate-fade-up divide-y divide-line/[.05]"
              >
                {pageRows.map((row) => (
                  <li key={row.id}>
                    <Link
                      to={`/recoveries/${row.id}`}
                      className="group grid grid-cols-2 items-center gap-x-3 gap-y-1.5 px-4 py-3 transition-colors hover:bg-surface-2/60 sm:grid-cols-[80px_1fr_120px_130px_110px_90px]"
                    >
                      <span className="flex items-center gap-1.5 font-mono text-[12px] text-ink-muted tnum">
                        #{row.id}
                        {row.id === VERIFIED_RECOVERY_EVENT_ID ? (
                          <ShieldCheck size={12} className="text-success" />
                        ) : null}
                      </span>
                      <span className="order-last col-span-2 flex items-center gap-2 text-[13px] text-ink sm:order-none sm:col-span-1">
                        {row.hasAgentRun ? (
                          <Bot size={13} className="shrink-0 text-accent" />
                        ) : (
                          <span className="size-3 shrink-0" />
                        )}
                        <span className="truncate">
                          {row.interventionCount > 0
                            ? `${row.interventionCount} intervention${
                                row.interventionCount > 1 ? "s" : ""
                              }`
                            : row.hasAgentRun
                              ? "Agent decision recorded"
                              : "Awaiting first intervention"}
                        </span>
                      </span>
                      <span className="text-[13px] font-medium text-ink tnum">
                        {rupeesFromPaise(row.amount_paise)}
                      </span>
                      <span className="text-2xs text-ink-muted">
                        {row.actions.length
                          ? actionLabel(row.actions[0]) +
                            (row.actions.length > 1 ? ` +${row.actions.length - 1}` : "")
                          : row.agentAction
                            ? actionLabel(row.agentAction)
                            : "—"}
                      </span>
                      <span>
                        <Badge
                          tone={
                            row.recovered
                              ? "success"
                              : row.status === "open"
                                ? "warning"
                                : "neutral"
                          }
                          dot
                        >
                          {row.recovered
                            ? "Recovered"
                            : row.status === "open"
                              ? "In recovery"
                              : "Not recovered"}
                        </Badge>
                      </span>
                      <span className="hidden text-right font-mono text-2xs text-ink-faint tnum sm:block">
                        {relTime(row.created_at)}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          {/* pagination */}
          {!isLoading && filtered.length > PAGE_SIZE ? (
            <div className="mt-4 flex items-center justify-between text-[13px] text-ink-muted">
              <span className="tnum">
                {clampedPage * PAGE_SIZE + 1}–
                {Math.min((clampedPage + 1) * PAGE_SIZE, filtered.length)} of{" "}
                {num(filtered.length)}
              </span>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={clampedPage === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                >
                  Previous
                </Button>
                <span className="tnum text-2xs text-ink-faint">
                  {clampedPage + 1} / {pageCount}
                </span>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={clampedPage >= pageCount - 1}
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                >
                  Next
                  <ArrowRight size={13} />
                </Button>
              </div>
            </div>
          ) : null}
        </>
      )}
    </Page>
  );
}
