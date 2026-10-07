"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { LogOut, Monitor, Moon, Settings, Sun } from "lucide-react";
import { useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { api, type User } from "@/lib/api";
import { buttonClass, cn } from "./ui";

export function BrandMark({ className }: { className?: string }) {
  return (
    <Link href="/" className={cn("group inline-flex items-center gap-2.5 text-[15px] font-semibold tracking-[-0.01em] text-fg", className)} aria-label="AI Interview Coach home">
      <span className="relative grid size-7 place-items-center overflow-hidden rounded-[9px] bg-[linear-gradient(120deg,var(--accent),var(--live))] shadow-[var(--highlight)]">
        <svg viewBox="0 0 24 24" className="size-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
          <path d="M5 12a7 7 0 0 1 14 0" />
          <path d="M8.5 12a3.5 3.5 0 0 1 7 0" />
          <circle cx="12" cy="12" r="0.9" fill="currentColor" />
        </svg>
      </span>
      <span>Interview Coach</span>
    </Link>
  );
}

type Theme = "system" | "light" | "dark";

function readTheme(): Theme {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(t: Theme) {
  const root = document.documentElement;
  if (t === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", t);
  try {
    if (t === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", t);
  } catch {
    /* storage unavailable: the choice lasts for this page only */
  }
  window.dispatchEvent(new Event("themechange"));
}

function subscribeTheme(cb: () => void) {
  window.addEventListener("themechange", cb);
  window.addEventListener("storage", cb);
  return () => { window.removeEventListener("themechange", cb); window.removeEventListener("storage", cb); };
}

/** The saved theme as React state (server render: "system"). */
export function useTheme(): Theme {
  return useSyncExternalStore(subscribeTheme, readTheme, () => "system");
}

export function ThemeToggle() {
  const theme = useTheme();
  const next: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
  const Icon = theme === "light" ? Sun : theme === "dark" ? Moon : Monitor;
  return (
    <button
      type="button"
      onClick={() => applyTheme(next[theme])}
      className={buttonClass("ghost", "sm", "w-8 px-0")}
      aria-label={`Theme: ${theme}. Switch to ${next[theme]}`}
      title={`Theme: ${theme}`}
    >
      <Icon className="size-4" />
    </button>
  );
}

/** Inline script for <head>: applies a saved theme before first paint (no flash). */
export const THEME_SCRIPT = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}`;

export function SiteHeader() {
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  useEffect(() => {
    api<User>("/auth/me").then(() => setSignedIn(true)).catch(() => setSignedIn(false));
  }, []);
  return (
    <header className="sticky top-0 z-40 border-b border-line/60 bg-canvas/75 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-5">
        <div className="flex items-center gap-8">
          <BrandMark />
          <nav className="hidden items-center gap-6 text-[13px] text-fg-muted md:flex" aria-label="Main">
            <Link href="/#how" className="hover:text-fg">How it works</Link>
            <Link href="/#proof" className="hover:text-fg">What we measure</Link>
            <Link href="/pricing" className="hover:text-fg">Pricing</Link>
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          {signedIn ? (
            <Link href="/dashboard" className={buttonClass("primary", "sm")}>Open app</Link>
          ) : (
            <>
              <Link href="/login" className={buttonClass("ghost", "sm")}>Sign in</Link>
              <Link href="/signup" className={buttonClass("primary", "sm")}>Start practising</Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export function AppHeader({ user }: { user: User | null }) {
  const path = usePathname();
  const router = useRouter();
  const links = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/interview/new", label: "New interview" },
    { href: "/settings", label: "Settings" },
  ];
  async function signOut() {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    router.replace("/login");
  }
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-canvas/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between gap-4 px-5">
        <div className="flex min-w-0 items-center gap-6">
          <BrandMark />
          <nav className="hidden items-center gap-1 sm:flex" aria-label="App">
            {links.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                aria-current={path?.startsWith(l.href) ? "page" : undefined}
                className={cn("rounded-[8px] px-3 py-1.5 text-[13px] font-medium transition-colors", path?.startsWith(l.href) ? "bg-surface-2 text-fg" : "text-fg-muted hover:text-fg")}
              >
                {l.label}
              </Link>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-1">
          <ThemeToggle />
          {user?.role === "admin" && <Link href="/admin" className={buttonClass("ghost", "sm")}>Admin</Link>}
          <Link href="/settings" className={buttonClass("ghost", "sm", "w-8 px-0 sm:hidden")} aria-label="Settings"><Settings className="size-4" /></Link>
          {user && (
            <span className="ml-2 hidden items-center gap-2 text-[13px] text-fg-muted md:flex">
              <span className="grid size-7 place-items-center rounded-full bg-surface-3 text-[12px] font-semibold text-fg">{(user.name || user.email)[0]?.toUpperCase()}</span>
            </span>
          )}
          <button onClick={signOut} className={buttonClass("ghost", "sm", "w-8 px-0")} aria-label="Sign out" title="Sign out"><LogOut className="size-4" /></button>
        </div>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1200px] flex-col gap-4 px-5 py-8 text-[12px] text-fg-subtle sm:flex-row sm:items-center sm:justify-between">
        <span>Patent pending (IN 202541122226). For interview practice, not for hiring decisions.</span>
        <nav className="flex gap-5" aria-label="Legal">
          <Link href="/legal/privacy" className="hover:text-fg">Privacy</Link>
          <Link href="/legal/terms" className="hover:text-fg">Terms</Link>
          <Link href="/pricing" className="hover:text-fg">Pricing</Link>
        </nav>
      </div>
    </footer>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-col gap-4 pb-8 sm:flex-row sm:items-end sm:justify-between">
      <div className="flex flex-col gap-1.5">
        <h1 className="text-[28px] font-semibold leading-9">{title}</h1>
        {description && <p className="max-w-2xl text-[14px] text-fg-muted">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
