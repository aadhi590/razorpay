import type { CSSProperties } from "react";
import { cn } from "@/lib/cn";

export function Skeleton({
  className,
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return <div className={cn("skeleton", className)} style={style} />;
}

export function SkeletonText({
  lines = 3,
  className,
}: {
  lines?: number;
  className?: string;
}) {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={cn("h-3", i === lines - 1 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

export function StatSkeleton() {
  return (
    <div className="panel p-5">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-3 h-9 w-32" />
      <Skeleton className="mt-3 h-3 w-20" />
    </div>
  );
}

export function CardSkeleton({ height = 240 }: { height?: number }) {
  return (
    <div className="panel p-5">
      <Skeleton className="h-3 w-32" />
      <Skeleton className="mt-4 w-full" style={{ height }} />
    </div>
  );
}
