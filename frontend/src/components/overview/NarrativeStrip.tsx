import {
  Radar,
  ScanSearch,
  TrendingUp,
  Sparkles,
  Zap,
  ShieldCheck,
  Flag,
} from "lucide-react";
import { cn } from "@/lib/cn";

const STEPS = [
  { label: "Detect", icon: Radar, note: "Failed payment enters recovery" },
  { label: "Understand", icon: ScanSearch, note: "Why the payment failed" },
  { label: "Predict", icon: TrendingUp, note: "ML + uplift score every action" },
  { label: "Decide", icon: Sparkles, note: "Agent picks the best next step" },
  { label: "Recover", icon: Zap, note: "Razorpay Payment Link" },
  { label: "Verify", icon: ShieldCheck, note: "Signed webhook confirms payment" },
  { label: "Stop", icon: Flag, note: "Agent ends the run" },
];

export function NarrativeStrip() {
  return (
    <div className="panel overflow-x-auto shadow-panel">
      <div className="flex min-w-[720px] items-stretch">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          const accent = i >= 2 && i <= 5;
          return (
            <div
              key={step.label}
              className="group relative flex flex-1 animate-fade-up flex-col gap-2 border-r border-line/[.07] px-4 py-4 last:border-r-0"
              style={{ animationDelay: `${i * 55}ms` }}
            >
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "grid size-7 place-items-center rounded-full ring-1 ring-inset",
                    accent
                      ? "bg-accent/10 text-accent ring-accent/25"
                      : "bg-surface-3 text-ink-muted ring-line/[.1]",
                  )}
                >
                  <Icon size={14} />
                </span>
                <span className="text-[11px] font-mono text-ink-faint tnum">
                  {String(i + 1).padStart(2, "0")}
                </span>
              </div>
              <div>
                <p className="text-[13px] font-semibold text-ink">{step.label}</p>
                <p className="mt-0.5 text-2xs leading-tight text-ink-faint">{step.note}</p>
              </div>
              {i < STEPS.length - 1 ? (
                <span className="pointer-events-none absolute -right-[7px] top-1/2 z-10 hidden -translate-y-1/2 text-ink-faint/40 sm:block">
                  ›
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
