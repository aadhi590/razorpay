import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import {
  Activity,
  FlaskConical,
  LayoutGrid,
  Menu,
  Moon,
  ScrollText,
  Sun,
  Wallet,
  X,
} from "lucide-react";
import { Logo } from "./Logo";
import { useTheme } from "../ThemeProvider";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/cn";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutGrid, end: true },
  { to: "/recoveries", label: "Recoveries", icon: Wallet, end: false },
  { to: "/analytics", label: "Analytics", icon: Activity, end: false },
  { to: "/experiments", label: "Experiments", icon: FlaskConical, end: false },
  { to: "/audit", label: "Audit", icon: ScrollText, end: false },
];

function HealthDot() {
  const { data, isError, isLoading } = useHealth();
  const ok = !isError && data?.status === "healthy";
  return (
    <div
      className="flex items-center gap-2 px-3 py-2 text-2xs"
      title={
        ok
          ? "Recovery engine online"
          : isLoading
            ? "Checking recovery engine…"
            : "Recovery engine unreachable"
      }
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          ok ? "bg-success" : isLoading ? "bg-warning" : "bg-danger",
          ok && "shadow-[0_0_0_3px_rgb(var(--success)/0.15)]",
        )}
      />
      <span className="font-medium text-ink-faint">
        {ok ? "Engine online" : isLoading ? "Connecting…" : "Engine offline"}
      </span>
    </div>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-col gap-0.5 px-3">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "group relative flex items-center gap-2.5 rounded-control px-3 py-2 text-[13px] font-medium transition-colors",
              isActive
                ? "bg-accent/[.1] text-ink ring-1 ring-inset ring-accent/25 before:absolute before:inset-y-1.5 before:-left-px before:w-0.5 before:rounded-full before:bg-accent before:content-['']"
                : "text-ink-muted hover:bg-surface-2 hover:text-ink",
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon
                size={16}
                className={cn(
                  "shrink-0 transition-colors",
                  isActive ? "text-accent" : "text-ink-faint group-hover:text-ink-muted",
                )}
              />
              {label}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      className="flex items-center gap-2.5 rounded-control px-3 py-2 text-[13px] font-medium text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
    >
      {theme === "dark" ? (
        <Sun size={16} className="text-ink-faint" />
      ) : (
        <Moon size={16} className="text-ink-faint" />
      )}
      {theme === "dark" ? "Light" : "Dark"} theme
    </button>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
      {/* desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-line/[.07] bg-surface/70 py-5 lg:flex">
        <div className="px-5 pb-5">
          <Logo />
          <p className="mt-1.5 text-2xs leading-tight text-ink-faint">
            AI Revenue Recovery
          </p>
        </div>
        <NavItems />
        <div className="mt-auto flex flex-col gap-0.5 px-3 pt-4">
          <ThemeToggle />
          <div className="mx-3 my-1 border-t border-line/[.07]" />
          <HealthDot />
        </div>
      </aside>

      {/* mobile header */}
      <header className="sticky top-0 z-40 flex items-center justify-between border-b border-line/[.07] bg-bg px-4 py-3 lg:hidden">
        <Logo />
        <button
          aria-label="Menu"
          onClick={() => setMobileOpen(true)}
          className="link-quiet -m-2 p-2"
        >
          <Menu size={20} />
        </button>
      </header>

      {mobileOpen ? (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setMobileOpen(false)}
          />
          <div className="absolute left-0 top-0 h-full w-72 border-r border-line/[.1] bg-surface py-5 shadow-pop">
            <div className="flex items-center justify-between px-5 pb-5">
              <Logo />
              <button
                aria-label="Close menu"
                onClick={() => setMobileOpen(false)}
                className="link-quiet -m-2 p-2"
              >
                <X size={18} />
              </button>
            </div>
            <NavItems onNavigate={() => setMobileOpen(false)} />
            <div className="mt-6 flex flex-col gap-0.5 px-3">
              <ThemeToggle />
              <HealthDot />
            </div>
          </div>
        </div>
      ) : null}

      <main className="min-w-0">{children}</main>
    </div>
  );
}

export function Page({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mx-auto w-full max-w-[1180px] px-5 py-7 sm:px-8 sm:py-9", className)}>
      {children}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-7 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        {eyebrow ? <div className="label-caps mb-1.5">{eyebrow}</div> : null}
        <h1 className="text-xl font-bold tracking-tight text-ink sm:text-[22px]">
          {title}
        </h1>
        {description ? (
          <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-ink-muted">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
