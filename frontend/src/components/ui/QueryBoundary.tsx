import type { ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { ErrorState } from "./States";
import { Skeleton } from "./Skeleton";

interface Props<T> {
  query: Pick<UseQueryResult<T>, "data" | "isLoading" | "isError" | "error" | "refetch">;
  children: (data: T) => ReactNode;
  loading?: ReactNode;
  skeletonHeight?: number;
  compactError?: boolean;
}

export function QueryBoundary<T>({
  query,
  children,
  loading,
  skeletonHeight = 160,
  compactError,
}: Props<T>) {
  if (query.isLoading) {
    return <>{loading ?? <Skeleton className="w-full" style={{ height: skeletonHeight }} />}</>;
  }
  if (query.isError) {
    return (
      <ErrorState
        error={query.error}
        onRetry={() => query.refetch()}
        compact={compactError}
      />
    );
  }
  if (query.data === undefined) {
    return (
      <ErrorState
        error={new Error("No data returned.")}
        onRetry={() => query.refetch()}
        compact={compactError}
      />
    );
  }
  return <>{children(query.data)}</>;
}
