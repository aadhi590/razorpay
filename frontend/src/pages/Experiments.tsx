import { FlaskConical } from "lucide-react";
import { Page, PageHeader } from "@/components/layout/Shell";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { QueryBoundary } from "@/components/ui/QueryBoundary";
import { EmptyState } from "@/components/ui/States";
import { Meter } from "@/components/ui/Bars";
import { useExperimentAnalytics, useExperimentList } from "@/lib/queries";
import type { ExperimentListItem, ExperimentResult } from "@/lib/types";
import { multiplier, num, pct, pp, rupeesCompactFromString } from "@/lib/format";

export default function Experiments() {
  const analytics = useExperimentAnalytics();
  const list = useExperimentList();

  return (
    <Page>
      <PageHeader
        eyebrow="Experimentation"
        title="Experiments"
        description="Each experiment holds out a randomised control arm so the lift attributed to an intervention is causal, not selection bias."
      />

      <QueryBoundary query={analytics} skeletonHeight={400}>
        {(d) => {
          const named = d.experiments.filter((e) => e.experiment_id != null);
          if (named.length === 0) {
            return (
              <Card>
                <EmptyState
                  icon={<FlaskConical size={18} />}
                  title="No experiments recorded"
                  description="Experiment results appear here once recovery events are assigned to an experiment."
                />
              </Card>
            );
          }
          const [lead, ...rest] = named;
          return (
            <div className="space-y-3">
              <ExperimentCard exp={lead} meta={list.data?.find((x) => x.id === lead.experiment_id)} lead />
              {rest.length > 0 ? (
                <div className="grid gap-3 lg:grid-cols-2">
                  {rest.map((exp) => (
                    <ExperimentCard
                      key={exp.experiment_id}
                      exp={exp}
                      meta={list.data?.find((x) => x.id === exp.experiment_id)}
                    />
                  ))}
                </div>
              ) : null}
            </div>
          );
        }}
      </QueryBoundary>
    </Page>
  );
}

function ExperimentCard({
  exp,
  meta,
  lead = false,
}: {
  exp: ExperimentResult;
  meta?: ExperimentListItem;
  lead?: boolean;
}) {
  const control = exp.variants.find((v) => v.variant === "control");
  const treatment = exp.variants.find((v) => v.variant === "treatment");
  const maxRate = Math.max(...exp.variants.map((v) => v.recovery_rate), 0.01);

  return (
    <Card className={lead ? "border-accent/20" : undefined}>
      <CardHeader
        eyebrow={`Experiment ${exp.experiment_id}`}
        title={exp.experiment_name ?? "Untitled experiment"}
        action={
          <div className="flex items-center gap-2">
            {meta?.status ? (
              <Badge tone={meta.status === "active" ? "success" : "neutral"} dot>
                {meta.status}
              </Badge>
            ) : null}
            {exp.relative_lift != null ? (
              <Badge tone="success">{multiplier(exp.relative_lift)} lift</Badge>
            ) : null}
          </div>
        }
      />
      <CardBody className="space-y-4">
        {lead && exp.absolute_lift != null ? (
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-control bg-accent/[.07] px-4 py-3 ring-1 ring-inset ring-accent/20">
            <span className="font-mono text-2xl font-bold text-success tnum">
              {pp(exp.absolute_lift)}
            </span>
            <span className="label-caps">absolute lift over control</span>
            {meta?.intervention_type ? (
              <span className="ml-auto text-2xs text-ink-faint">
                intervention: {meta.intervention_type.replace(/_/g, " ")}
              </span>
            ) : null}
          </div>
        ) : null}

        <div className={lead ? "grid gap-4 sm:grid-cols-2" : "space-y-4"}>
          {[control, treatment].filter(Boolean).map((v) => (
            <div key={v!.variant}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[13px] font-medium capitalize text-ink">
                  {v!.variant === "treatment" ? "AI-driven" : "Control"}
                </span>
                <span className="font-mono text-[13px] text-ink tnum">
                  {pct(v!.recovery_rate)}
                </span>
              </div>
              <Meter
                value={v!.recovery_rate}
                max={maxRate}
                tone={v!.variant === "treatment" ? "accent" : "neutral"}
              />
              <p className="mt-1.5 text-2xs text-ink-faint">
                {num(v!.recovered_events)} of {num(v!.recovery_events)} recovered ·{" "}
                {rupeesCompactFromString(v!.recovered_value)}
              </p>
            </div>
          ))}
        </div>

        {!lead && exp.absolute_lift != null ? (
          <div className="flex items-center gap-2 border-t border-line/[.07] pt-3 text-[13px]">
            <span className="label-caps">Absolute lift</span>
            <span className="font-mono font-semibold text-success tnum">
              {pp(exp.absolute_lift)}
            </span>
            {meta?.intervention_type ? (
              <span className="ml-auto text-2xs text-ink-faint">
                intervention: {meta.intervention_type.replace(/_/g, " ")}
              </span>
            ) : null}
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}
