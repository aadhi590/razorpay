import { ArrowUpRight, Check, CircleDashed, ShieldCheck } from "lucide-react";
import type { RazorpayInterventionView, RazorpayConfigStatus } from "@/lib/types";
import { Card, CardBody, CardHeader } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { CopyButton } from "../ui/CopyButton";
import { rupeesFromPaise, truncateId, actionLabel } from "@/lib/format";
import { cn } from "@/lib/cn";

const FLOW = ["Link created", "Awaiting payment", "Payment confirmed", "Recovered"];

function flowIndex(iv: RazorpayInterventionView): number {
  if (iv.outcome_payment_recovered) return 3;
  if (iv.payment_link_paid) return 2;
  if (iv.payment_link_created) return 1;
  return 0;
}

export function RazorpayCard({
  interventions,
  config,
  amountPaise,
}: {
  interventions: RazorpayInterventionView[];
  config: RazorpayConfigStatus;
  amountPaise: number;
}) {
  const linked = interventions.filter((iv) => iv.payment_link_created);
  const primary = linked[0];

  return (
    <Card>
      <CardHeader
        eyebrow="Payment infrastructure"
        title="Razorpay recovery"
        action={
          config.test_mode ? (
            <Badge tone="accent" dot>
              Test Mode
            </Badge>
          ) : null
        }
      />
      <CardBody className="space-y-4">
        {primary ? (
          <>
            {/* state flow */}
            <div className="flex items-center gap-1.5">
              {FLOW.map((label, i) => {
                const active = i <= flowIndex(primary);
                const isRecovered = i === 3 && flowIndex(primary) === 3;
                return (
                  <div key={label} className="flex flex-1 flex-col items-center gap-1.5">
                    <div className="flex w-full items-center">
                      <span
                        className={cn(
                          "grid size-5 shrink-0 place-items-center rounded-full text-[10px] ring-1 ring-inset transition-colors",
                          active
                            ? isRecovered
                              ? "bg-success/15 text-success ring-success/40"
                              : "bg-accent/12 text-accent ring-accent/35"
                            : "bg-surface-3 text-ink-faint ring-line/[.1]",
                        )}
                      >
                        {active ? <Check size={11} strokeWidth={3} /> : i + 1}
                      </span>
                      {i < FLOW.length - 1 ? (
                        <span
                          className={cn(
                            "h-px flex-1",
                            i < flowIndex(primary) ? "bg-accent/40" : "bg-line/[.1]",
                          )}
                        />
                      ) : null}
                    </div>
                    <span
                      className={cn(
                        "text-center text-[10px] leading-tight",
                        active ? "text-ink-muted" : "text-ink-faint",
                      )}
                    >
                      {label}
                    </span>
                  </div>
                );
              })}
            </div>

            {primary.outcome_payment_recovered ? (
              <div className="flex animate-fade-up items-center gap-3 rounded-control border border-success/25 bg-success/[.07] p-3.5">
                <span className="grid size-9 shrink-0 place-items-center rounded-full bg-success/15 text-success">
                  <ShieldCheck size={18} />
                </span>
                <div>
                  <p className="text-[15px] font-semibold text-ink tnum">
                    {rupeesFromPaise(primary.recovered_amount_paise ?? amountPaise, true)}{" "}
                    recovered
                  </p>
                  <p className="text-2xs text-ink-muted">
                    Payment confirmed by a signature-verified Razorpay webhook.
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 rounded-control border border-line/[.08] bg-surface-2 p-3.5">
                <span className="grid size-9 shrink-0 place-items-center rounded-full bg-surface-3 text-ink-faint">
                  <CircleDashed size={18} />
                </span>
                <div>
                  <p className="text-[13px] font-semibold text-ink">
                    Payment Link created — not yet paid
                  </p>
                  <p className="text-2xs text-ink-muted">
                    Recovery stays <span className="font-medium">unconfirmed</span> until a
                    verified <code className="font-mono">payment_link.paid</code> webhook
                    arrives.
                  </p>
                </div>
              </div>
            )}

            {/* details grid */}
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-[13px]">
              <Detail label="Action">{actionLabel(primary.action_type)}</Detail>
              <Detail label="Link status">
                <span className="capitalize">{primary.last_razorpay_status ?? "—"}</span>
              </Detail>
              <Detail label="Amount">{rupeesFromPaise(amountPaise, true)}</Detail>
              <Detail label="Amount paid">
                {primary.payment_link_paid
                  ? rupeesFromPaise(primary.recovered_amount_paise ?? amountPaise, true)
                  : "₹0.00"}
              </Detail>
              <Detail label="Payment Link ID">
                <span className="inline-flex items-center gap-1.5">
                  <span className="font-mono text-[12px]">
                    {truncateId(primary.razorpay_payment_link_id)}
                  </span>
                  {primary.razorpay_payment_link_id ? (
                    <CopyButton value={primary.razorpay_payment_link_id} label="" />
                  ) : null}
                </span>
              </Detail>
              <Detail label="Razorpay Payment ID">
                {primary.razorpay_payment_id ? (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="font-mono text-[12px]">
                      {truncateId(primary.razorpay_payment_id)}
                    </span>
                    <CopyButton value={primary.razorpay_payment_id} label="" />
                  </span>
                ) : (
                  <span className="text-ink-faint">—</span>
                )}
              </Detail>
              <Detail label="Reference ID" full>
                <span className="font-mono text-[12px] text-ink-muted">
                  {primary.razorpay_reference_id ?? "—"}
                </span>
              </Detail>
            </dl>

            {primary.razorpay_short_url ? (
              <a
                href={primary.razorpay_short_url}
                target="_blank"
                rel="noreferrer noopener"
                className="inline-flex items-center gap-1.5 text-[13px] font-medium text-accent hover:underline"
              >
                Open Payment Link
                <ArrowUpRight size={14} />
              </a>
            ) : null}
          </>
        ) : (
          <div className="rounded-control border border-line/[.08] bg-surface-2 p-4 text-[13px] text-ink-muted">
            This recovery ran a direct intervention —{" "}
            <span className="font-medium text-ink">no Razorpay Payment Link</span> was
            created for it. The Razorpay execution path is exercised by the verified
            recovery and the live agent run.
          </div>
        )}

        <p className="border-t border-line/[.07] pt-3 text-2xs leading-relaxed text-ink-faint">
          Test Mode only. Amount and currency come from the authoritative payment
          record, never from the model. No API keys or webhook secrets are exposed.
        </p>
      </CardBody>
    </Card>
  );
}

function Detail({
  label,
  children,
  full,
}: {
  label: string;
  children: React.ReactNode;
  full?: boolean;
}) {
  return (
    <div className={full ? "col-span-2" : undefined}>
      <dt className="label-caps mb-0.5">{label}</dt>
      <dd className="text-ink">{children}</dd>
    </div>
  );
}
