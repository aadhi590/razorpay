import type { ReactNode } from "react";
import { AlertTriangle, Inbox, RefreshCw, WifiOff } from "lucide-react";
import { ApiError, NetworkError } from "@/lib/api";
import { Button } from "./Button";
import { cn } from "@/lib/cn";

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 px-6 py-14 text-center",
        className,
      )}
    >
      <div className="grid size-11 place-items-center rounded-full bg-surface-2 text-ink-faint ring-1 ring-inset ring-line/[.08]">
        {icon ?? <Inbox size={18} />}
      </div>
      <div className="max-w-sm space-y-1">
        <p className="text-sm font-semibold text-ink">{title}</p>
        {description ? (
          <p className="text-[13px] leading-relaxed text-ink-muted">{description}</p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

interface ErrorStateProps {
  error: unknown;
  onRetry?: () => void;
  compact?: boolean;
  className?: string;
}

function describe(error: unknown): { title: string; body: string; icon: ReactNode } {
  if (error instanceof NetworkError) {
    return {
      icon: <WifiOff size={18} />,
      title: "Can't reach the recovery engine",
      body: "The API isn't responding. Start the backend (uvicorn app.main:app) and check VITE_API_BASE_URL. Your recovery data is safe.",
    };
  }
  if (error instanceof ApiError) {
    if (error.status === 404)
      return { icon: <AlertTriangle size={18} />, title: "Not found", body: error.detail };
    if (error.status === 503)
      return {
        icon: <AlertTriangle size={18} />,
        title: "Service not configured",
        body: error.detail || "A required integration isn't configured on the backend.",
      };
    if (error.status >= 500)
      return {
        icon: <AlertTriangle size={18} />,
        title: "The recovery engine hit an error",
        body: "The request failed server-side. Nothing was changed — retry in a moment.",
      };
    return {
      icon: <AlertTriangle size={18} />,
      title: "Request rejected",
      body: error.detail,
    };
  }
  return {
    icon: <AlertTriangle size={18} />,
    title: "Something went wrong",
    body: error instanceof Error ? error.message : "An unexpected error occurred.",
  };
}

export function ErrorState({ error, onRetry, compact, className }: ErrorStateProps) {
  const d = describe(error);
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 text-center",
        compact ? "px-4 py-8" : "px-6 py-14",
        className,
      )}
    >
      <div className="grid size-11 place-items-center rounded-full bg-danger/10 text-danger ring-1 ring-inset ring-danger/20">
        {d.icon}
      </div>
      <div className="max-w-md space-y-1">
        <p className="text-sm font-semibold text-ink">{d.title}</p>
        <p className="text-[13px] leading-relaxed text-ink-muted">{d.body}</p>
      </div>
      {onRetry ? (
        <Button size="sm" variant="secondary" onClick={onRetry}>
          <RefreshCw size={13} />
          Try again
        </Button>
      ) : null}
    </div>
  );
}
