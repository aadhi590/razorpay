import { useEffect, useRef, useState } from "react";

/** Count a number up once on mount. Respects prefers-reduced-motion. */
export function useCountUp(target: number, durationMs = 650, enabled = true): number {
  const [value, setValue] = useState(enabled ? 0 : target);
  const raf = useRef<number>();
  useEffect(() => {
    if (!enabled) {
      setValue(target);
      return;
    }
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !Number.isFinite(target)) {
      setValue(target);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [target, durationMs, enabled]);
  return value;
}
