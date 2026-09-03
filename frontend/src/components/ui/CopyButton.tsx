import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/cn";

export function CopyButton({
  value,
  label,
  className,
}: {
  value: string;
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1400);
        } catch {
          /* clipboard unavailable */
        }
      }}
      aria-label={copied ? "Copied" : `Copy ${label ?? "value"}`}
      className={cn(
        "inline-flex items-center gap-1.5 text-ink-faint transition-colors hover:text-ink",
        className,
      )}
    >
      {copied ? (
        <Check size={13} className="text-success" />
      ) : (
        <Copy size={13} />
      )}
      {label ? <span className="text-2xs font-medium">{copied ? "Copied" : label}</span> : null}
    </button>
  );
}

export function Mono({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span className={cn("font-mono text-[12px] text-ink-muted tnum", className)}>
      {children}
    </span>
  );
}
