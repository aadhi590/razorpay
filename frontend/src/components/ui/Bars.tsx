import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";

type Tone = "accent" | "success" | "warning" | "danger" | "neutral";
const FILL: Record<Tone, string> = {
  accent: "bg-accent",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
  neutral: "bg-ink-faint",
};

/** Flip to true one frame after mount so CSS transitions run once. */
function useGrow(): boolean {
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return grown;
}

/** A single horizontal meter. `value` and `max` in any consistent unit. */
export function Meter({
  value,
  max,
  tone = "accent",
  className,
}: {
  value: number;
  max: number;
  tone?: Tone;
  className?: string;
}) {
  const grown = useGrow();
  const pct = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-surface-3", className)}>
      <div
        className={cn(
          "h-full origin-left rounded-full transition-transform duration-500 ease-out",
          FILL[tone],
        )}
        style={{ transform: `scaleX(${grown ? pct : 0})` }}
      />
    </div>
  );
}

export interface RankedItem {
  label: string;
  value: number;
  display: string;
  tone?: Tone;
  highlight?: boolean;
  caption?: string;
}

/** Ranked horizontal bars, best-first. Used for intervention intelligence. */
export function RankedBars({
  items,
  max,
  className,
}: {
  items: RankedItem[];
  max?: number;
  className?: string;
}) {
  const grown = useGrow();
  const ceiling = max ?? Math.max(...items.map((i) => i.value), 1);
  return (
    <div className={cn("space-y-3", className)}>
      {items.map((item, i) => (
        <div key={item.label} className="grid grid-cols-[minmax(0,1fr)] gap-1.5">
          <div className="flex items-center justify-between gap-3">
            <span
              className={cn(
                "flex items-center gap-2 text-[13px]",
                item.highlight ? "font-semibold text-ink" : "text-ink-muted",
              )}
            >
              {item.highlight ? (
                <span className="size-1.5 rounded-full bg-accent" aria-hidden />
              ) : null}
              {item.label}
            </span>
            <span
              className={cn(
                "font-mono text-[12px] tnum",
                item.highlight ? "text-ink" : "text-ink-faint",
              )}
            >
              {item.display}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-surface-3">
            <div
              className={cn(
                "h-full origin-left rounded-full transition-transform duration-500 ease-out",
                FILL[item.tone ?? (item.highlight ? "accent" : "neutral")],
                !item.highlight && "opacity-60",
              )}
              style={{
                transform: `scaleX(${grown ? Math.max(0.02, item.value / ceiling) : 0})`,
                transitionDelay: `${i * 45}ms`,
              }}
            />
          </div>
          {item.caption ? (
            <span className="text-2xs text-ink-faint">{item.caption}</span>
          ) : null}
        </div>
      ))}
    </div>
  );
}
