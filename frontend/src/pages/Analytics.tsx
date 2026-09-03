import { Page, PageHeader } from "@/components/layout/Shell";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Stat } from "@/components/ui/Stat";
import { Badge } from "@/components/ui/Badge";
import { Skeleton, StatSkeleton } from "@/components/ui/Skeleton";
import { QueryBoundary } from "@/components/ui/QueryBoundary";
import { ErrorState } from "@/components/ui/States";
import { ActionRateChart, ReliabilityBars } from "@/components/charts/Charts";
import { PortfolioAllocation } from "@/components/analytics/PortfolioAllocation";
import {
  useRecoveryImpact,
  useControlVsTreatment,
  useActionAnalytics,
  useMlModel,
  useUpliftModel,
} from "@/lib/queries";
import {
  actionLabel,
  multiplier,
  num,
  pct,
  pp,
  rupeesCompact,
  rupeesCompactFromString,
  rupeesFromString,
} from "@/lib/format";

interface ReliabilityRow {
  bin: string;
  mean_predicted: number;
  observed_rate: number;
}

export default function Analytics() {
  const impact = useRecoveryImpact();
  const cvt = useControlVsTreatment();
  const actions = useActionAnalytics();
  const ml = useMlModel();
  const uplift = useUpliftModel();

  return (
    <Page>
      <PageHeader
        eyebrow="Measurement"
        title="Analytics"
        description="Money recovered above a randomised control baseline, what drives it, and how the models behind the decisions actually perform."
      />

      {/* impact hero */}
      {impact.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <StatSkeleton key={i} />
          ))}
        </div>
      ) : impact.isError ? (
        <Card>
          <CardBody>
            <ErrorState error={impact.error} onRetry={() => impact.refetch()} />
          </CardBody>
        </Card>
      ) : impact.data && impact.data.computable ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Incremental revenue recovered"
              value={rupeesCompact(
                (impact.data.incremental_revenue_recovered_paise ?? 0) / 100,
              )}
              sub="vs control baseline"
              tone="success"
              hero
              className="sm:col-span-2"
            />
            <Stat
              label="Incremental recovery rate"
              value={pp(impact.data.incremental_recovery_rate)}
              sub="treated − control"
              tone="accent"
            />
            <Stat
              label="Treated / control"
              value={`${num(impact.data.treated_group_size)} / ${num(
                impact.data.control_group_size,
              )}`}
              sub={`${num(impact.data.recovered_treated_events)} vs ${num(
                impact.data.recovered_control_events,
              )} recovered`}
            />
          </div>

          {impact.data.incremental_recovery_rate_ci_95 ? (
            <div className="mt-3 flex flex-col gap-2 rounded-control bg-accent/[.08] p-3.5 ring-1 ring-inset ring-accent/25 sm:flex-row sm:items-center sm:gap-4">
              <div className="shrink-0">
                <div className="label-caps mb-1 text-accent/90">
                  95% confidence interval
                </div>
                <div className="font-mono text-[17px] font-bold text-accent tnum">
                  {pct(impact.data.incremental_recovery_rate_ci_95[0])} –{" "}
                  {pct(impact.data.incremental_recovery_rate_ci_95[1])}
                </div>
              </div>
              <Badge
                tone={
                  impact.data.incremental_recovery_rate_ci_95[0] > 0 ||
                  impact.data.incremental_recovery_rate_ci_95[1] < 0
                    ? "success"
                    : "warning"
                }
                dot
              >
                {impact.data.incremental_recovery_rate_ci_95[0] > 0 ||
                impact.data.incremental_recovery_rate_ci_95[1] < 0
                  ? "Excludes zero — a real, measured effect"
                  : "Includes zero — directional only"}
              </Badge>
              <p className="text-2xs leading-relaxed text-ink-muted sm:ml-auto sm:max-w-md">
                {impact.data.confidence_note}
              </p>
            </div>
          ) : null}
        </>
      ) : (
        <Card>
          <CardBody>
            <p className="text-[13px] text-ink-muted">
              {impact.data?.reason ?? "Impact is not computable for the current data."}
            </p>
          </CardBody>
        </Card>
      )}

      {/* control vs treatment + actions */}
      <div className="mt-8 grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader
            eyebrow="Randomised comparison"
            title="Control vs AI-driven"
            action={
              cvt.data?.relative_lift != null ? (
                <Badge tone="success">{multiplier(cvt.data.relative_lift)}</Badge>
              ) : null
            }
          />
          <CardBody>
            <QueryBoundary query={cvt} skeletonHeight={220}>
              {(d) => (
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="label-caps [&>th]:pb-2 [&>th]:text-left">
                      <th>Group</th>
                      <th className="text-right">Events</th>
                      <th className="text-right">Recovery rate</th>
                      <th className="text-right">Recovered</th>
                    </tr>
                  </thead>
                  <tbody className="tnum [&>tr>td]:border-t [&>tr>td]:border-line/[.06] [&>tr>td]:py-2.5">
                    <tr>
                      <td className="text-ink-muted">Control</td>
                      <td className="text-right">{num(d.control.recovery_events)}</td>
                      <td className="text-right">{pct(d.control.recovery_rate)}</td>
                      <td className="text-right">
                        {rupeesCompactFromString(d.control.recovered_value)}
                      </td>
                    </tr>
                    <tr>
                      <td className="font-medium text-accent">AI-driven</td>
                      <td className="text-right">{num(d.treatment.recovery_events)}</td>
                      <td className="text-right font-medium text-accent">
                        {pct(d.treatment.recovery_rate)}
                      </td>
                      <td className="text-right">
                        {rupeesCompactFromString(d.treatment.recovered_value)}
                      </td>
                    </tr>
                    <tr>
                      <td className="font-medium text-success">Lift</td>
                      <td className="text-right text-ink-faint">—</td>
                      <td className="text-right font-medium text-success">
                        {pp(d.absolute_lift)}
                      </td>
                      <td className="text-right text-ink-faint">—</td>
                    </tr>
                  </tbody>
                </table>
              )}
            </QueryBoundary>
          </CardBody>
        </Card>

        <Card>
          <CardHeader eyebrow="By action" title="Intervention performance" />
          <CardBody>
            <QueryBoundary query={actions} skeletonHeight={220}>
              {(d) => (
                <>
                  <ActionRateChart actions={d.actions} />
                  <div className="mt-3 space-y-1.5 border-t border-line/[.07] pt-3 text-2xs">
                    {[...d.actions]
                      .sort((a, b) => b.recovery_rate - a.recovery_rate)
                      .map((a) => (
                        <div
                          key={a.action_type}
                          className="flex items-center justify-between text-ink-muted"
                        >
                          <span>{actionLabel(a.action_type)}</span>
                          <span className="font-mono tnum">
                            {num(a.interventions)} runs ·{" "}
                            {a.cost_per_recovery
                              ? `${rupeesFromString(a.cost_per_recovery, true)}/recovery`
                              : "—"}
                          </span>
                        </div>
                      ))}
                  </div>
                </>
              )}
            </QueryBoundary>
          </CardBody>
        </Card>
      </div>

      {/* portfolio scarcity allocation */}
      <h2 className="mb-3 mt-8 flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.08em] text-ink-muted">
        <span className="h-3.5 w-0.5 rounded-full bg-accent/60" aria-hidden />
        Batch allocation
      </h2>
      <p className="mb-3 max-w-2xl text-[13px] leading-relaxed text-ink-muted">
        The agent decides one event at a time. This ranks every currently-open
        eligible event by expected value and shows what a limited number of
        interventions can and cannot cover — and what that limit costs.
      </p>
      <PortfolioAllocation />

      {/* model intelligence */}
      <h2 className="mb-3 mt-8 flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.08em] text-ink-muted">
        <span className="h-3.5 w-0.5 rounded-full bg-accent/60" aria-hidden />
        Model intelligence
      </h2>
      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader
            eyebrow="Recovery-response model"
            title="ML — probability of recovery"
            action={
              ml.data?.model_version ? (
                <Badge tone="neutral">{ml.data.model_version}</Badge>
              ) : null
            }
          />
          <CardBody className="space-y-4">
            {ml.isLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : ml.isError ? (
              <ErrorState error={ml.error} onRetry={() => ml.refetch()} compact />
            ) : ml.data?.available ? (
              <>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[13px]">
                  <MetaRow label="Algorithm" value={ml.data.algorithm ?? "—"} />
                  <MetaRow
                    label="Test ROC-AUC"
                    value={metricStr(ml.data.test_metrics, ["classification", "roc_auc"])}
                  />
                  <MetaRow
                    label="Test PR-AUC"
                    value={metricStr(ml.data.test_metrics, ["classification", "pr_auc"])}
                  />
                  <MetaRow
                    label="Calibration error"
                    value={metricStr(ml.data.calibration, ["test_calibrated"], true)}
                  />
                </dl>
                {reliabilityRows(ml.data.test_metrics).length ? (
                  <div>
                    <p className="label-caps mb-2">Calibration — predicted vs observed</p>
                    <ReliabilityBars rows={reliabilityRows(ml.data.test_metrics)} />
                  </div>
                ) : null}
                {ml.data.synthetic_benchmark ? (
                  <p className="text-2xs text-ink-faint">
                    Synthetic benchmark — trained on generated data with a known outcome
                    process. Illustrative, not production.
                  </p>
                ) : null}
              </>
            ) : (
              <p className="text-[13px] text-ink-muted">{ml.data?.detail ?? "Model unavailable."}</p>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            eyebrow="Causal model"
            title="Uplift — incremental effect"
            action={
              uplift.data?.model_version ? (
                <Badge tone="neutral">{uplift.data.model_version}</Badge>
              ) : null
            }
          />
          <CardBody className="space-y-4">
            {uplift.isLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : uplift.isError ? (
              <ErrorState error={uplift.error} onRetry={() => uplift.refetch()} compact />
            ) : uplift.data?.available ? (
              <>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[13px]">
                  <MetaRow label="Learner" value={uplift.data.learner_type ?? "—"} />
                  <MetaRow label="Base algorithm" value={uplift.data.base_algorithm ?? "—"} />
                  <MetaRow
                    label="Qini coefficient"
                    value={metricStr(uplift.data.test_evaluation, ["qini", "qini_coefficient"], true)}
                  />
                  <MetaRow
                    label="Policy value gain"
                    value={metricStr(
                      uplift.data.test_evaluation,
                      ["policy_value", "uplift_policy", "gain_vs_random_action"],
                      true,
                    )}
                  />
                </dl>
                {uplift.data.limitations?.length ? (
                  <div>
                    <p className="label-caps mb-2">Stated limitations</p>
                    <ul className="space-y-1.5 text-2xs leading-relaxed text-ink-muted">
                      {uplift.data.limitations.slice(0, 4).map((l, i) => (
                        <li key={i} className="flex gap-1.5">
                          <span className="mt-1.5 size-1 shrink-0 rounded-full bg-ink-faint" />
                          {l}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </>
            ) : (
              <p className="text-[13px] text-ink-muted">
                {uplift.data?.detail ?? "Model unavailable."}
              </p>
            )}
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label-caps mb-0.5">{label}</dt>
      <dd className="font-mono text-ink tnum">{value}</dd>
    </div>
  );
}

function metricStr(
  obj: Record<string, unknown> | null | undefined,
  path: string[],
  round4 = false,
): string {
  let cur: unknown = obj;
  for (const k of path) {
    if (cur && typeof cur === "object" && k in cur) {
      cur = (cur as Record<string, unknown>)[k];
    } else {
      return "—";
    }
  }
  if (typeof cur === "number") return round4 ? cur.toFixed(4) : cur.toFixed(3);
  return cur == null ? "—" : String(cur);
}

function reliabilityRows(
  metrics: Record<string, unknown> | null | undefined,
): ReliabilityRow[] {
  const table =
    ((metrics?.calibration as Record<string, unknown> | undefined)
      ?.reliability_table as unknown[] | undefined) ?? [];
  return table
    .filter((r): r is Record<string, number> => !!r && typeof r === "object")
    .map((r) => ({
      bin: String(r.bin ?? ""),
      mean_predicted: Number(r.mean_predicted ?? 0),
      observed_rate: Number(r.observed_rate ?? 0),
    }));
}
