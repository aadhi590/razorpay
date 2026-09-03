import { cn } from "@/lib/cn";

export function Logo({ className, size = 22 }: { className?: string; size?: number }) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        aria-hidden
        className="shrink-0"
      >
        <rect width="32" height="32" rx="7" className="fill-accent" />
        <path
          d="M9 20c0-5 3-9 8-9 3 0 5 2 5 4s-2 3-4 3H12"
          className="stroke-bg"
          strokeWidth="2.4"
          strokeLinecap="round"
        />
        <path
          d="M15 21l-3 3 3 3"
          className="stroke-bg"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="text-[15px] font-semibold tracking-tight text-ink">
        Reclaim
      </span>
    </span>
  );
}
