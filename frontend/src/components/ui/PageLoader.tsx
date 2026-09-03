import { Page } from "../layout/Shell";
import { Skeleton, StatSkeleton } from "./Skeleton";

export function PageLoader() {
  return (
    <Page>
      <Skeleton className="h-4 w-24" />
      <Skeleton className="mt-2 h-7 w-64" />
      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <StatSkeleton key={i} />
        ))}
      </div>
      <Skeleton className="mt-3 h-[280px] w-full" />
    </Page>
  );
}
