import { Link } from "react-router-dom";
import { ArrowRight, ShieldCheck, Sparkles } from "lucide-react";
import { Page, PageHeader } from "@/components/layout/Shell";
import { Card, CardBody, CardHeader, SectionTitle } from "@/components/ui/Card";
import { Stat } from "@/components/ui/Stat";
import { useCountUp } from "@/lib/useCountUp";
import { StatSkeleton, Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/States";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { NarrativeStrip } from "@/components/overview/NarrativeStrip";
import { FeaturedRecoveries } from "@/components/overview/FeaturedRecoveries";
import { LiftChart, EventStateDonut } from "@/components/charts/Charts";
import {
  useSummary,
  useControlVsTreatment,
  useRecoveryImpact,
  useActionAnalytics,
  useAuditLogList,
} from "@/lib/queries";
import { VERIFIED_RECOVERY_EVENT_ID } from "@/lib/config";
import {
  actionLabel,
  multiplier,
  num,
  pct,
  pp,
  relTime,
  rupeesCompact,
  rupeesCompactFromString,
  titleCase,
} from "@/lib/format";

function HeroRecovered({ rupees }: { rupees: number }) {
  const v = useCountUp(rupees);
  return <>{rupeesCompact(v)}</>;
}

export default function Overview() {
  const summary = useSummary();
  const cvt = useControlVsTreatment();
  const impact = useRecoveryImpact();
  const actions = useActionAnalytics();
  const audit = useAuditLogList();

  const s = summary.data;
  const recoveredRupees = s ? Number(s.total_recovered_value) : 0;
  const atRiskRupees = s ? Number(s.total_failed_payment_value) : 0;

  return (
    <Page>
      <PageHeader
        eyebrow="Command center"
        title="AI Revenue Recovery"
        description="Failed payments aren't lost revenue — they're revenue at risk. Reclaim understands why each one failed and recovers it intelligently."
        actions={
          <Link to={`/recoveries/${VERIFIED_RECOVERY_EVENT_ID}`}>
            <Button variant="primary" size="sm">
              <ShieldCheck size={14} />
              Open verified recovery
            </Button>
          </Link>
        }
      />

      {/* ---- HERO KPIs ------------------------------------------------ */}
      {summary.isError ? (
        <Card>
          <CardBody>
            <ErrorState error={summary.error} onRetry={() => summary.refetch()} />
          </CardBody>
        </Card>
      ) : summary.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <StatSkeleton key={i} />
          ))}
        </div>
      ) : s ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="panel-hero flex flex-col justify-between p-5 sm:col-span-2 lg:col-span-2">
            <div className="label-caps">Revenue recovered</div>
            <div className="mt-2 font-bold leading-none tracking-tight text-success tnum text-[clamp(2.4rem,6vw,3.6rem)]">
              <HeroRecovered rupees={recoveredRupees} />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
              {impact.data?.incremental_revenue_recovered_paise != null ? (
                <span className="font-medium text-success">
                  {rupeesCompact(impact.data.incremental_revenue_recovered_paise / 100)}{" "}
                  <span className="font-normal text-ink-muted">
                    attributable to the AI vs a randomised control baseline
                  </span>
                </span>
              ) : (
                <span className="text-ink-muted">
                  across {num(s.recovered_events)} recovered payments
                </span>
              )}
            </div>
            {impact.data?.computable && impact.data.incremental_recovery_rate_ci_95 ? (
              <div className="mt-4 flex flex-wrap items-baseline gap-x-2 gap-y-1 rounded-control bg-accent/[.09] px-3 py-2.5 ring-1 ring-inset ring-accent/25">
                <span className="text-lg font-bold tabular-nums text-accent">
                  {pp(impact.data.incremental_recovery_rate)}
                </span>
                <span className="text-2xs font-semibold uppercase tracking-[0.06em] text-accent/90">
                  incremental recovery
                </span>
                <span className="text-2xs tabular-nums text-ink-muted">
                  · 95% CI {pct(impact.data.incremental_recovery_rate_ci_95[0])}–
                  {pct(impact.data.incremental_recovery_rate_ci_95[1])} · measured,
                  not modelled
                </span>
              </div>
            ) : null}
          </div>

          <Stat
            label="Revenue at risk"
            value={rupeesCompact(atRiskRupees)}
            sub={`${num(s.total_failed_payments)} failed payments`}
            tone="warning"
          />
          <Stat
            label="Recovery rate"
            value={pct(s.overall_recovery_rate)}
            delta={
              cvt.data
                ? { value: pp(cvt.data.absolute_lift), positive: cvt.data.absolute_lift > 0 }
                : null
            }
            sub={cvt.data ? "vs control" : `${num(s.recovered_events)} of ${num(s.total_recovery_events)}`}
            tone="accent"
          />
        </div>
      ) : null}

      <div className="mt-3">
        <SectionTitle>How Reclaim recovers a payment</SectionTitle>
        <NarrativeStrip />
      </div>

      {/* ---- performance ------------------------------------------- */}
      <div className="mt-8 grid gap-3 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            eyebrow="Causal impact"
            title="AI-driven recovery vs randomised control"
            action={
              cvt.data?.relative_lift != null ? (
                <Badge tone="success">{multiplier(cvt.data.relative_lift)} lift</Badge>
              ) : null
            }
          />
          <CardBody>
            {cvt.isLoading ? (
              <Skeleton className="h-[180px] w-full" />
            ) : cvt.isError ? (
              <ErrorState error={cvt.error} onRetry={() => cvt.refetch()} compact />
            ) : cvt.data ? (
              <>
                <LiftChart
                  control={cvt.data.control.recovery_rate}
                  treatment={cvt.data.treatment.recovery_rate}
                />
                <div className="mt-3 grid grid-cols-3 gap-3 border-t border-line/[.07] pt-3 text-[13px]">
                  <div>
                    <div className="label-caps mb-1">Control</div>
                    <div className="font-semibold tnum">
                      {pct(cvt.data.control.recovery_rate)}
                    </div>
                    <div className="text-2xs text-ink-faint">
                      {num(cvt.data.control.recovery_events)} events
                    </div>
                  </div>
                  <div>
                    <div className="label-caps mb-1">AI-driven</div>
                    <div className="font-semibold text-accent tnum">
                      {pct(cvt.data.treatment.recovery_rate)}
                    </div>
                    <div className="text-2xs text-ink-faint">
                      {num(cvt.data.treatment.recovery_events)} events
                    </div>
                  </div>
                  <div>
                    <div className="label-caps mb-1">Absolute lift</div>
                    <div className="font-semibold text-success tnum">
                      {pp(cvt.data.absolute_lift)}
                    </div>
                    {impact.data?.incremental_recovery_rate_ci_95 ? (
                      <div className="text-2xs text-ink-faint">
                        95% CI {pct(impact.data.incremental_recovery_rate_ci_95[0])}–
                        {pct(impact.data.incremental_recovery_rate_ci_95[1])}
                      </div>
                    ) : null}
                  </div>
                </div>
              </>
            ) : null}
          </CardBody>
        </Card>

        <Card>
          <CardHeader eyebrow="Portfolio" title="Recovery events" />
          <CardBody>
            {summary.isLoading ? (
              <Skeleton className="h-[180px] w-full" />
            ) : s ? (
              <>
                <EventStateDonut
                  recovered={s.recovered_events}
                  open={s.open_events}
                  abandoned={s.abandoned_events}
                />
                <ul className="mt-2 space-y-1.5 text-[13px]">
                  <LegendRow color="bg-success" label="Recovered" value={num(s.recovered_events)} />
                  <LegendRow color="bg-warning" label="In recovery" value={num(s.open_events)} />
                  <LegendRow
                    color="bg-ink-faint"
                    label="Not recovered"
                    value={num(s.abandoned_events)}
                  />
                </ul>
              </>
            ) : null}
          </CardBody>
        </Card>
      </div>

      {/* ---- intelligence + interventions ------------------------- */}
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <FeaturedRecoveries />

        <Card>
          <CardHeader
            eyebrow="What works"
            title="Intervention performance"
            action={
              <Link
                to="/analytics"
                className="inline-flex items-center gap-1 text-2xs font-medium text-ink-muted hover:text-ink"
              >
                Analytics <ArrowRight size={12} />
              </Link>
            }
          />
          <CardBody>
            {actions.isLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : actions.isError ? (
              <ErrorState error={actions.error} onRetry={() => actions.refetch()} compact />
            ) : actions.data ? (
              <ul className="space-y-3">
                {[...actions.data.actions]
                  .sort((a, b) => b.recovery_rate - a.recovery_rate)
                  .map((a) => {
                    const max = Math.max(
                      ...actions.data!.actions.map((x) => x.recovery_rate),
                    );
                    return (
                      <li key={a.action_type}>
                        <div className="flex items-center justify-between text-[13px]">
                          <span className="text-ink">{actionLabel(a.action_type)}</span>
                          <span className="font-mono text-ink-muted tnum">
                            {pct(a.recovery_rate)}
                          </span>
                        </div>
                        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                          <div
                            className="h-full rounded-full bg-accent"
                            style={{ width: `${(a.recovery_rate / max) * 100}%` }}
                          />
                        </div>
                        <div className="mt-1 text-2xs text-ink-faint">
                          {rupeesCompactFromString(a.recovered_value)} recovered ·{" "}
                          {a.cost_per_recovery
                            ? `${rupeesCompactFromString(a.cost_per_recovery)} per recovery`
                            : "—"}
                        </div>
                      </li>
                    );
                  })}
              </ul>
            ) : null}
          </CardBody>
        </Card>
      </div>

      {/* ---- recent audit --------------------------------------- */}
      <div className="mt-3">
        <Card>
          <CardHeader
            eyebrow="Trust"
            title="Recent activity"
            action={
              <Link
                to="/audit"
                className="inline-flex items-center gap-1 text-2xs font-medium text-ink-muted hover:text-ink"
              >
                Full audit trail <ArrowRight size={12} />
              </Link>
            }
          />
          <CardBody className="p-0">
            {audit.isLoading ? (
              <div className="space-y-2 p-5">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-8 w-full" />
                ))}
              </div>
            ) : audit.isError ? (
              <div className="p-5">
                <ErrorState error={audit.error} onRetry={() => audit.refetch()} compact />
              </div>
            ) : audit.data ? (
              <ul className="divide-y divide-line/[.06]">
                {audit.data.slice(-7).reverse().map((row) => (
                  <li
                    key={row.id}
                    className="flex items-center gap-3 px-5 py-2.5 text-[13px]"
                  >
                    <Badge tone="neutral">{titleCase(row.actor)}</Badge>
                    <span className="min-w-0 flex-1 truncate text-ink-muted">
                      {row.action.replace(/_/g, " ")}
                    </span>
                    <span className="shrink-0 font-mono text-2xs text-ink-faint tnum">
                      {relTime(row.created_at)}
                    </span>
                  </li>
                ))}
              </ul>
            ) : null}
          </CardBody>
        </Card>
      </div>

      <p className="mt-6 flex items-center justify-center gap-1.5 text-2xs text-ink-faint">
        <Sparkles size={12} />
        Every figure on this page is returned by the backend. Nothing is
        estimated or illustrative in the UI.
      </p>
    </Page>
  );
}

function LegendRow({
  color,
  label,
  value,
}: {
  color: string;
  label: string;
  value: string;
}) {
  return (
    <li className="flex items-center gap-2">
      <span className={`size-2 rounded-[2px] ${color}`} />
      <span className="flex-1 text-ink-muted">{label}</span>
      <span className="font-mono text-ink tnum">{value}</span>
    </li>
  );
}
