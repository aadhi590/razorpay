/** Presentation helpers. Never mutate data — format only. */

const INR = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});
const INR2 = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const NUM = new Intl.NumberFormat("en-IN");

/** paise (integer) → "₹1,23,456" */
export function rupeesFromPaise(paise: number | null | undefined, decimals = false): string {
  if (paise == null || Number.isNaN(paise)) return "—";
  const v = paise / 100;
  return decimals ? INR2.format(v) : INR.format(v);
}

/** backend Decimal rupee string (e.g. "958420.00") → "₹9,58,420" */
export function rupeesFromString(s: string | null | undefined, decimals = false): string {
  if (s == null) return "—";
  const v = Number(s);
  if (Number.isNaN(v)) return "—";
  return decimals ? INR2.format(v) : INR.format(v);
}

/** Compact Indian currency for hero figures: ₹9.58L, ₹81.05L, ₹1.2Cr */
export function rupeesCompact(value: number): string {
  if (Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e7) return `₹${(value / 1e7).toFixed(2)} Cr`;
  if (abs >= 1e5) return `₹${(value / 1e5).toFixed(2)} L`;
  if (abs >= 1e3) return `₹${(value / 1e3).toFixed(1)} K`;
  return `₹${Math.round(value)}`;
}
export function rupeesCompactFromPaise(paise: number | null | undefined): string {
  if (paise == null) return "—";
  return rupeesCompact(paise / 100);
}
export function rupeesCompactFromString(s: string | null | undefined): string {
  if (s == null) return "—";
  const v = Number(s);
  return Number.isNaN(v) ? "—" : rupeesCompact(v);
}

export function num(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? "—" : NUM.format(n);
}

/** ratio 0..1 → "14.3%" */
export function pct(ratio: number | null | undefined, digits = 1): string {
  if (ratio == null || Number.isNaN(ratio)) return "—";
  return `${(ratio * 100).toFixed(digits)}%`;
}
/** already-percentage-points value → "+11.7pp" */
export function pp(ratioDiff: number | null | undefined, digits = 1): string {
  if (ratioDiff == null || Number.isNaN(ratioDiff)) return "—";
  const v = ratioDiff * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}pp`;
}
export function multiplier(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${x.toFixed(digits)}×`;
}

export function relTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const diff = Date.now() - then;
  const s = Math.round(diff / 1000);
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  if (d < 30) return `${d}d ago`;
  const mo = Math.round(d / 30);
  if (mo < 12) return `${mo}mo ago`;
  return `${Math.round(mo / 12)}y ago`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
export function timeOnly(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function ms(n: number | null | undefined): string {
  if (n == null) return "—";
  return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${Math.round(n)}ms`;
}

export function truncateId(id: string | null | undefined, head = 10, tail = 4): string {
  if (!id) return "—";
  if (id.length <= head + tail + 1) return id;
  return `${id.slice(0, head)}…${id.slice(-tail)}`;
}

/** action_type → display label */
export const ACTION_LABELS: Record<string, string> = {
  retry: "Smart retry",
  sms_nudge: "SMS nudge",
  whatsapp_nudge: "WhatsApp nudge",
  method_switch_prompt: "Payment method switch",
};
export function actionLabel(a: string | null | undefined): string {
  if (!a) return "—";
  return ACTION_LABELS[a] ?? a.replace(/_/g, " ");
}

export const FAILURE_LABELS: Record<string, string> = {
  insufficient_funds: "Insufficient funds",
  card_expired: "Card expired",
  bank_timeout: "Bank timeout",
  issuer_decline: "Issuer decline",
};
export function failureLabel(f: string | null | undefined): string {
  if (!f) return "Unknown";
  return FAILURE_LABELS[f] ?? f.replace(/_/g, " ");
}

export function titleCase(s: string): string {
  return s.replace(/(^|[\s_-])([a-z])/g, (_, p, c) => p.replace(/[_-]/, " ") + c.toUpperCase());
}
