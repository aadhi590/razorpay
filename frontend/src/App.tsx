import { Suspense, lazy, useEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { Shell } from "./components/layout/Shell";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { PageLoader } from "./components/ui/PageLoader";

const Overview = lazy(() => import("./pages/Overview"));
const Recoveries = lazy(() => import("./pages/Recoveries"));
const RecoveryDetail = lazy(() => import("./pages/RecoveryDetail"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Experiments = lazy(() => import("./pages/Experiments"));
const Audit = lazy(() => import("./pages/Audit"));
const NotFound = lazy(() => import("./pages/NotFound"));

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <Shell>
      <ScrollToTop />
      <ErrorBoundary>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/recoveries" element={<Recoveries />} />
            <Route path="/recoveries/:id" element={<RecoveryDetail />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/audit" element={<Audit />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </Shell>
  );
}
