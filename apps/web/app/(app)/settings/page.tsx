"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowRight, CreditCard, Fingerprint, LogOut, ShieldCheck, Activity } from "lucide-react";
import { Alert, Badge, Button, Card, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

export default function Settings() {
  // useUser loads GET /api/auth/me (api<User>("/auth/me")) and redirects to /login when signed out.
  const { user } = useUser();
  const router = useRouter();
  const [signingOut, setSigningOut] = useState(false);
  const [error, setError] = useState("");

  async function signOut() {
    setSigningOut(true);
    setError("");
    try {
      await api("/auth/logout", { method: "POST" });
      router.push("/");
    } catch {
      setError("Could not sign out. Please try again.");
      setSigningOut(false);
    }
  }

  const sections = [
    { href: "/settings/privacy", icon: ShieldCheck, title: "Privacy and data", body: "Permissions, download your data, delete face and voice templates or your account." },
    { href: "/settings/billing", icon: CreditCard, title: "Plan and billing", body: "Your current plan, sessions used this month and upgrades." },
    { href: "/enroll", icon: Fingerprint, title: "Identity checks", body: "Optional face and voice checks that confirm the same person stays in the session." },
    ...(user?.role === "admin" ? [{ href: "/admin", icon: Activity, title: "Operations", body: "Cost, model quality and latency across the deployment." }] : []),
  ];

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-8">
      <header>
        <p className="text-[13px] font-medium text-accent">Settings</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9">Account</h1>
      </header>

      <Card as="section" aria-labelledby="profile-heading" className="flex flex-col gap-5 p-5">
        <h2 id="profile-heading" className="text-[16px] font-semibold">Profile</h2>
        <dl className="grid gap-5 sm:grid-cols-3">
          <div className="flex min-w-0 flex-col gap-1">
            <dt className="text-[12px] font-medium uppercase tracking-[0.06em] text-fg-subtle">Name</dt>
            <dd className="truncate text-[15px] text-fg">{user ? user.name || "Not set" : <Skeleton className="h-5 w-32" />}</dd>
          </div>
          <div className="flex min-w-0 flex-col gap-1">
            <dt className="text-[12px] font-medium uppercase tracking-[0.06em] text-fg-subtle">Email</dt>
            <dd className="truncate text-[15px] text-fg">{user ? user.email : <Skeleton className="h-5 w-48" />}</dd>
          </div>
          <div className="flex min-w-0 flex-col gap-1">
            <dt className="text-[12px] font-medium uppercase tracking-[0.06em] text-fg-subtle">Plan</dt>
            <dd className="flex items-center gap-2 text-[15px] text-fg">
              {user ? (
                <>
                  <Badge tone={user.plan === "free" ? "neutral" : "accent"} className="capitalize">{user.plan}</Badge>
                  {user.plan === "free" && <Link href="/settings/billing" className="text-[13px] text-fg-muted hover:text-fg">Upgrade</Link>}
                </>
              ) : <Skeleton className="h-5 w-16" />}
            </dd>
          </div>
        </dl>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {sections.map((s) => (
          <Link
            key={s.href}
            href={s.href}
            className="group flex items-start gap-4 rounded-[16px] border border-line bg-surface p-5 shadow-[var(--highlight)] transition-colors duration-[120ms] hover:border-line-strong hover:bg-surface-2"
          >
            <span className="flex size-10 shrink-0 items-center justify-center rounded-[12px] border border-line bg-surface-2 text-accent">
              <s.icon className="size-5" aria-hidden />
            </span>
            <span className="flex min-w-0 flex-1 flex-col gap-1">
              <span className="text-[15px] font-semibold text-fg">{s.title}</span>
              <span className="text-[13px] leading-5 text-fg-muted">{s.body}</span>
            </span>
            <ArrowRight className="mt-1 size-4 shrink-0 text-fg-subtle transition-transform duration-[120ms] group-hover:translate-x-0.5" aria-hidden />
          </Link>
        ))}
      </div>

      <Card as="section" aria-labelledby="session-heading" className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 id="session-heading" className="text-[16px] font-semibold">Sign out</h2>
          <p className="mt-1 text-[13px] text-fg-muted">End your session on this device.</p>
        </div>
        <Button variant="secondary" onClick={() => void signOut()} loading={signingOut} disabled={!user}>
          {!signingOut && <LogOut className="size-4" aria-hidden />} Sign out
        </Button>
      </Card>
      {error && <Alert tone="danger">{error}</Alert>}
    </div>
  );
}
