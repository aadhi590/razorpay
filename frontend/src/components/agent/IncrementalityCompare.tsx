import { ArrowRight } from "lucide-react";
import type { ActionIncrementality } from "@/lib/types";
import { Badge, type Tone } from "../ui/Badge";
import { pp } from "@/lib/format";

/**
 * The predicted-vs-observed comparison for one action, rendered inline on the
 * `get_historical_incrementality_for_action` trace row. Uses only existing state
 * colours (success = the model's prediction lands inside the observed 95% CI,
 * warning = it lands outside it, neutral = no prediction to compare against) and
 * existing spacing/type tokens.
 */
const DIVERGENCE_PP = 0.025; // fallback when no CI is available

export function IncrementalityCompare({ data }: { data: ActionIncrementality }) {
  if (!data.computable) {
    return (
      <div className="mt-2 rounded-[7px] border border-line/[.08] bg-surface-2 p-2.5 text-2xs leading-relaxed text-ink-muted">
        <span className="font-medium text-ink-faint">Not enough history.</span>{" "}
        {data.note}
      </div>
    );
  }

  const observed = data.observed_incremental_lift ?? 0;
  const predicted = data.model_predicted_uplift_for_context;
  const ci = data.observed_incremental_lift_ci_95;

  let verdict: { tone: Tone; text: string };
  if (predicted == null) {
    verdict = { tone: "neutral", text: "No model prediction to compare" };
  } else if (ci) {
    if (predicted >= ci[0] && predicted <= ci[1]) {
      verdict = { tone: "success", text: "Prediction lands inside the observed range" };
    } else if (predicted > ci[1]) {
      verdict = { tone: "warning", text: "Model is more optimistic than history" };
    } else {
      verdict = { tone: "warning", text: "History beats the model's prediction" };
    }
  } else if (Math.abs(predicted - observed) < DIVERGENCE_PP) {
    verdict = { tone: "success", text: "Predicted and observed agree" };
  } else if (predicted > observed) {
    verdict = { tone: "warning", text: "Model is more optimistic than history" };
  } else {
    verdict = { tone: "warning", text: "History beats the model's prediction" };
  }

  return (
    <div className="mt-2 rounded-[7px] border border-line/[.08] bg-surface-2 p-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {predicted != null ? (
          <>
            <Figure label="Model predicted" value={pp(predicted)} muted />
            <ArrowRight size={13} className="shrink-0 text-ink-faint" aria-hidden />
          </>
        ) : null}
        <Figure
          label="Actually observed"
          value={pp(observed)}
          sub={
            ci
              ? `95% CI ${pp(ci[0])} to ${pp(ci[1])}`
              : undefined
          }
        />
        <span className="ml-auto">
          <Badge tone={verdict.tone} dot>
            {verdict.text}
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
