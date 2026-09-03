import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, PALETTE } from "./chartkit";
import { actionLabel, pct, rupeesCompact } from "@/lib/format";

function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string; color?: string; dataKey?: string }>;
  label?: ReactNode;
  formatter?: (value: number | string, name: string) => ReactNode;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="panel px-3 py-2 shadow-pop">
      {label != null ? (
        <p className="mb-1 text-2xs font-semibold text-ink">{label}</p>
      ) : null}
      <div className="space-y-0.5">
        {payload.map((p, i) => (
          <div key={i} className="flex items-center gap-2 text-2xs">
            <span
              className="size-2 rounded-[2px]"
              style={{ background: p.color ?? PALETTE.accent }}
            />
            <span className="text-ink-muted">{p.name ?? p.dataKey}</span>
            <span className="ml-auto font-mono text-ink tnum">
              {formatter
                ? formatter(p.value ?? "", String(p.name ?? p.dataKey ?? ""))
                : String(p.value ?? "")}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Control vs treatment recovery rate. */
export function LiftChart({
  control,
  treatment,
}: {
  control: number;
  treatment: number;
}) {
  const data = [
    { name: "Control", rate: control, fill: PALETTE.faint },
    { name: "AI-driven", rate: treatment, fill: PALETTE.accent },
  ];
  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
        <XAxis type="number" hide domain={[0, Math.max(treatment, control) * 1.25]} />
        <YAxis type="category" dataKey="name" width={78} {...AXIS} />
        <Tooltip
          cursor={{ fill: "rgb(var(--line) / 0.05)" }}
          content={<ChartTooltip formatter={(v) => pct(Number(v))} />}
        />
        <Bar dataKey="rate" name="Recovery rate" radius={[4, 4, 4, 4]} barSize={22}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Per-action recovery rate. */
export function ActionRateChart({
  actions,
}: {
  actions: { action_type: string; recovery_rate: number; recovered_value: string }[];
}) {
  const data = [...actions]
    .sort((a, b) => b.recovery_rate - a.recovery_rate)
    .map((a) => ({
      name: actionLabel(a.action_type),
      rate: a.recovery_rate,
      value: Number(a.recovered_value),
    }));
  return (
    <ResponsiveContainer width="100%" height={210}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
        <XAxis type="number" hide domain={[0, "dataMax"]} />
        <YAxis type="category" dataKey="name" width={150} {...AXIS} />
        <Tooltip
          cursor={{ fill: "rgb(var(--line) / 0.05)" }}
          content={
            <ChartTooltip
              formatter={(v, name) =>
                name === "Recovered" ? rupeesCompact(Number(v)) : pct(Number(v))
              }
            />
          }
        />
        <Bar dataKey="rate" name="Recovery rate" fill={PALETTE.accent} radius={[0, 3, 3, 0]} barSize={16} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Event-state donut: recovered / in recovery / not recovered. */
export function EventStateDonut({
  recovered,
  open,
  abandoned,
}: {
  recovered: number;
  open: number;
  abandoned: number;
}) {
  const data = [
    { name: "Recovered", value: recovered, fill: PALETTE.success },
    { name: "In recovery", value: open, fill: PALETTE.warning },
    { name: "Not recovered", value: abandoned, fill: PALETTE.faint },
  ];
  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius={52}
          outerRadius={74}
          paddingAngle={2}
          stroke="none"
        >
          {data.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );
}

/** Generic small calibration / reliability line as bars (predicted vs observed). */
export function ReliabilityBars({
  rows,
}: {
  rows: { bin: string; mean_predicted: number; observed_rate: number }[];
}) {
  const data = rows.map((r) => ({
    name: r.bin,
    predicted: r.mean_predicted,
    observed: r.observed_rate,
  }));
  return (
    <ResponsiveContainer width="100%" height={190}>
      <BarChart data={data} margin={{ left: -12, right: 8, top: 8, bottom: 4 }}>
        <XAxis dataKey="name" {...AXIS} interval={0} angle={0} />
        <YAxis {...AXIS} tickFormatter={(v) => pct(Number(v), 0)} width={40} />
        <Tooltip
          cursor={{ fill: "rgb(var(--line) / 0.05)" }}
          content={<ChartTooltip formatter={(v) => pct(Number(v))} />}
        />
        <Bar dataKey="predicted" name="Predicted" fill={PALETTE.faint} radius={[3, 3, 0, 0]} barSize={14} />
        <Bar dataKey="observed" name="Observed" fill={PALETTE.accent} radius={[3, 3, 0, 0]} barSize={14} />
      </BarChart>
    </ResponsiveContainer>
  );
}
