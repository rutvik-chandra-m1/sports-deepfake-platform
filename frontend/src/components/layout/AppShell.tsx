import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/upload", label: "Upload" },
  { to: "/history", label: "History" },
];

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex min-h-svh flex-col">
      <header className="border-b border-border bg-bg-raised">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-md border border-accent/40 bg-accent/10">
              <span className="h-2 w-2 rounded-full bg-accent" />
            </div>
            <div className="flex flex-col leading-none">
              <span className="font-display text-sm font-semibold tracking-wide text-text">
                REPLAY<span className="text-accent">/</span>VERIFY
              </span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-text-faint">
                Sports Deepfake Verification
              </span>
            </div>
          </div>

          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-md px-3 py-2 text-sm transition-colors ${
                    isActive
                      ? "bg-surface text-text"
                      : "text-text-muted hover:bg-surface hover:text-text"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>

      <footer className="border-t border-border px-6 py-4">
        <p className="mx-auto max-w-6xl font-mono text-xs text-text-faint">
          Final Year Engineering Major Project — built incrementally, milestone by milestone.
        </p>
      </footer>
    </div>
  );
}
