import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { pp } from "@/lib/format";

/**
 * The Analytics page's dominant element: the Newcombe/Wilson 95% CI for the
 * incremental recovery rate, drawn as a horizontal interval plotted against a
 * zero marker. Distinct in shape from Overview's single big-number hero — this
 * is a measured *range*, and seeing it sit clear of zero is the whole point.
 *
 * Motion: the interval segment sweeps out from its lower bound once on mount
 * using the established bar-grow vocabulary (scaleX, 500ms ease-out, origin-left)
 * — the same transition Bars.tsx uses. Fully neutralised by the global
 * prefers-reduced-motion rule (transition-duration → 0.001ms), leaving the bar
 * at its correct final width.
 */
export function ConfidenceIntervalBand({
  point,
  ci,
  note,
}: {
  point: number;
  ci: [number, number];
  note?: string | null;
}) {
  const [lo, hi] = ci;
  const excludesZero = lo > 0 || hi < 0;

  // Domain always includes zero so the exclusion gap is visible.
  const min = Math.min(0, lo);
  const max = Math.max(0, hi);
  const pad = (max - min) * 0.12 || 0.01;
  const domainLo = min - pad;
  const domainHi = max + pad;
  const span = domainHi - domainLo;
  const at = (v: number) => ((v - domainLo) / span) * 100;

  const zeroPct = at(0);
  const loPct = at(lo);
  const hiPct = at(hi);
  const pointPct = at(point);

  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div className="panel-hero p-5 sm:p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="label-caps text-accent/90">
            Incremental recovery rate · 95% confidence interval
          </div>
          <div className="mt-1.5 flex items-baseline gap-2.5">
            <span className="font-mono text-[clamp(1.8rem,4vw,2.6rem)] font-bold leading-none text-accent tnum">
              {pp(lo)} – {pp(hi)}
            </span>
          </div>
          <div className="mt-1.5 text-[13px] text-ink-muted">
            point estimate{" "}
            <span className="font-mono font-semibold text-ink tnum">{pp(point)}</span> ·
            Newcombe/Wilson, not a p-value
          </div>
        </div>
        <Badge tone={excludesZero ? "success" : "warning"} dot>
          {excludesZero
            ? "Excludes zero — a real, measured effect"
            : "Includes zero — directional only"}
        </Badge>
      </div>

      {/* interval plot */}
      <div className="mt-6">
        <div className="relative h-9">
          {/* baseline track */}
          <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line/[.14]" />

          {/* zero marker */}
          <div
            className="absolute bottom-0 top-0 w-px bg-line/[.35]"
            style={{ left: `${zeroPct}%` }}
          />

          {/* the CI segment — scales out from its lower bound */}
          <div
            className="absolute top-1/2 h-2 -translate-y-1/2 origin-left rounded-full bg-accent/70 transition-transform duration-500 ease-out"
            style={{
              left: `${loPct}%`,
              width: `${Math.max(hiPct - loPct, 0.5)}%`,
              transform: `scaleX(${grown ? 1 : 0})`,
            }}
          />
          {/* bound caps */}
          <div
            className="absolute top-1/2 h-4 w-px -translate-y-1/2 bg-accent transition-opacity duration-500 ease-out"
            style={{ left: `${loPct}%`, opacity: grown ? 1 : 0 }}
          />
          <div
            className="absolute top-1/2 h-4 w-px -translate-y-1/2 bg-accent transition-opacity duration-500 ease-out"
            style={{ left: `${hiPct}%`, opacity: grown ? 1 : 0 }}
          />
          {/* point estimate */}
          <div
            className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent ring-2 ring-surface transition-opacity duration-500 ease-out"
            style={{ left: `${pointPct}%`, opacity: grown ? 1 : 0 }}
          />
        </div>
        <div className="relative mt-2 h-4 font-mono text-2xs text-ink-faint tnum">
          <span className="absolute left-0">{pp(domainLo)}</span>
          <span
            className="absolute -translate-x-1/2 text-ink-muted"
            style={{ left: `${zeroPct}%` }}
          >
            0
          </span>
          <span className="absolute right-0">{pp(domainHi)}</span>
        </div>
      </div>

      {note ? (
        <p className="mt-4 max-w-3xl border-t border-line/[.08] pt-3 text-2xs leading-relaxed text-ink-muted">
          {note}
        </p>
      ) : null}
    </div>
  );
}
