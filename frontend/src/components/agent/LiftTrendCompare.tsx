import { ArrowRight } from "lucide-react";
import type { ActionLiftTrend } from "@/lib/types";
import { Badge, type Tone } from "../ui/Badge";
import { pp } from "@/lib/format";

/**
 * The recent-vs-earlier effectiveness trend for one action, rendered inline on
 * the `get_action_lift_trend` trace row — the sibling of IncrementalityCompare,
 * using the same compact inline layout, the same `Figure` grammar, and only
 * existing Badge tones:
 *   improving  -> success   (getting better in the field)
 *   declining  -> danger    (eroding — the same strong-negative tone States/Stat use)
 *   stable / insufficient -> neutral (no trend distinguishable from noise)
 *
 * The confidence interval is shown in the compact inline style (matching
 * IncrementalityCompare, the directly analogous element), not the bordered
 * Analytics-hero style, because it sits inside a dense trace row.
 */
const TONE: Record<ActionLiftTrend["trend_direction"], Tone> = {
  improving: "success",
  declining: "danger",
  stable_or_insufficient_data: "neutral",
};
const VERDICT: Record<ActionLiftTrend["trend_direction"], string> = {
  improving: "Effectiveness improving over time",
  declining: "Effectiveness declining over time",
  stable_or_insufficient_data: "No trend — flat within noise",
};

export function LiftTrendCompare({ data }: { data: ActionLiftTrend }) {
  if (!data.computable) {
    return (
      <div className="mt-2 rounded-[7px] border border-line/[.08] bg-surface-2 p-2.5 text-2xs leading-relaxed text-ink-muted">
        <span className="font-medium text-ink-faint">
          Not enough {data.reason === "insufficient_baseline_data" ? "earlier" : "recent"} data.
        </span>{" "}
        {data.note}
      </div>
    );
  }

  const ci = data.trend_confidence_interval;

  return (
    <div className="mt-2 rounded-[7px] border border-line/[.08] bg-surface-2 p-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Figure
          label="Earlier window"
          value={data.baseline_window_lift != null ? pp(data.baseline_window_lift) : "—"}
          sub={`${data.baseline_window_size} uses`}
          muted
        />
        <ArrowRight size={13} className="shrink-0 text-ink-faint" aria-hidden />
        <Figure
          label="Recent window"
          value={data.recent_window_lift != null ? pp(data.recent_window_lift) : "—"}
          sub={
            ci
              ? `change 95% CI ${pp(ci[0])} to ${pp(ci[1])}`
              : `${data.recent_window_size} uses`
          }
        />
        <span className="ml-auto">
          <Badge tone={TONE[data.trend_direction]} dot>
            {VERDICT[data.trend_direction]}
          </Badge>
        </span>
      </div>
      <p className="mt-2.5 border-t border-line/[.07] pt-2 text-2xs leading-relaxed text-ink-faint">
        {data.note}
      </p>
    </div>
  );
}

function Figure({
  label,
  value,
  sub,
  muted,
}: {
  label: string;
  value: string;
  sub?: string;
  muted?: boolean;
}) {
  return (
    <div>
      <div className="label-caps">{label}</div>
      <div
        className={
          "mt-0.5 font-mono text-[15px] font-semibold tnum " +
          (muted ? "text-ink-muted" : "text-ink")
        }
      >
        {value}
      </div>
      {sub ? <div className="mt-0.5 text-2xs text-ink-faint">{sub}</div> : null}
    </div>
  );
}
