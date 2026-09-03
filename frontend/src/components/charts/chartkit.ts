/** Shared Recharts styling tokens — one visual language for every chart. */
export const AXIS = {
  stroke: "transparent",
  tick: { fill: "rgb(var(--ink-faint))", fontSize: 11 },
  tickLine: false,
  axisLine: false,
} as const;

export const PALETTE = {
  accent: "rgb(var(--accent))",
  success: "rgb(var(--success))",
  warning: "rgb(var(--warning))",
  danger: "rgb(var(--danger))",
  faint: "rgb(var(--ink-faint))",
  track: "rgb(var(--surface-3))",
};
