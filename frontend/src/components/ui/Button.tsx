import { forwardRef, type ButtonHTMLAttributes } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:brightness-110 active:brightness-95 shadow-[0_1px_0_0_rgb(255_255_255_/_0.12)_inset]",
  secondary:
    "bg-surface-3 text-ink hover:bg-surface-2 ring-1 ring-inset ring-line/[.12]",
  ghost: "text-ink-muted hover:text-ink hover:bg-surface-2",
  danger: "bg-danger/12 text-danger hover:bg-danger/20 ring-1 ring-inset ring-danger/25",
};
const SIZES: Record<Size, string> = {
  sm: "h-8 px-3 text-xs gap-1.5 rounded-[7px]",
  md: "h-9 px-3.5 text-[13px] gap-2 rounded-control",
};

export const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  { variant = "secondary", size = "md", loading, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "inline-flex select-none items-center justify-center font-medium transition-[background,filter,color] duration-150",
        "disabled:cursor-not-allowed disabled:opacity-50",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {loading ? <Loader2 size={size === "sm" ? 13 : 15} className="animate-spin" /> : null}
      {children}
    </button>
  );
});
