/**
 * Runtime configuration. The API base URL is the only thing the frontend needs,
 * and it is a public URL — never a secret. When VITE_API_BASE_URL is unset we
 * fall back to a same-origin "/api" path (the Vite dev proxy handles it).
 */
const raw = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();

export const API_BASE_URL = raw && raw.length > 0 ? raw.replace(/\/$/, "") : "";

/** The real, Razorpay Test Mode–verified recovery event. Pinned in the UI. */
export const VERIFIED_RECOVERY_EVENT_ID = 18499;

/** Rows to show per page in the recoveries explorer. */
export const PAGE_SIZE = 25;
