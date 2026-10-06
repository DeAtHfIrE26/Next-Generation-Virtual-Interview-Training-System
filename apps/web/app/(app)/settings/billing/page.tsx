"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft, CreditCard, Globe, IndianRupee } from "lucide-react";
import { Alert, Badge, Button, Card, ProgressBar, Skeleton, Stat } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

type Status = { plan: string; sessions_this_month: number; sessions_per_month: number;
  subscription: { provider: string; status: string; plan: string } | null; providers: { stripe: boolean; razorpay: boolean } };

export default function Billing() {
  const { user } = useUser();
  const [st, setSt] = useState<Status | null>(null);
  const [msg, setMsg] = useState("");
  const [pending, setPending] = useState<"stripe" | "razorpay" | null>(null);
  useEffect(() => { if (user) void api<Status>("/billing/status").then(setSt); }, [user]);

  async function go(provider: "stripe" | "razorpay") {
    setPending(provider);
    setMsg("");
    try {
      const r = await api<{ url: string }>(provider === "stripe" ? "/billing/stripe/checkout" : "/billing/razorpay/subscription",
        { method: "POST", json: { plan: "pro" } });
      window.location.href = r.url;
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Could not start checkout.");
      setPending(null);
    }
  }

  const used = st ? (st.sessions_per_month > 0 ? st.sessions_this_month / st.sessions_per_month : 0) : 0;

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-8">
      <header>
        <Link href="/settings" className="mb-3 inline-flex items-center gap-1 text-[13px] text-fg-muted hover:text-fg"><ArrowLeft className="size-3.5" aria-hidden /> Settings</Link>
        <p className="text-[13px] font-medium text-accent">Settings</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9">Plan and billing</h1>
      </header>

      {!user || !st ? (
        <div className="flex max-w-[720px] flex-col gap-4" aria-busy="true">
          <Skeleton className="h-36" />
          <Skeleton className="h-28" />
        </div>
      ) : (
        <div className="flex max-w-[720px] flex-col gap-6">
          <Card as="section" aria-labelledby="plan-heading" className="flex flex-col gap-5 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 id="plan-heading" className="flex items-center gap-2 text-[16px] font-semibold"><CreditCard className="size-4 text-accent" aria-hidden /> Current plan</h2>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={st.plan === "free" ? "neutral" : "accent"} className="capitalize">{st.plan}</Badge>
                {st.subscription && <Badge tone="live">{st.subscription.provider} subscription {st.subscription.status}</Badge>}
              </div>
            </div>
            <div className="flex flex-col gap-3">
              <Stat label="Sessions this month" value={`${st.sessions_this_month} of ${st.sessions_per_month}`} />
              <ProgressBar value={used} tone={used >= 1 ? "warn" : "accent"} label={`Sessions used this month: ${st.sessions_this_month} of ${st.sessions_per_month}`} />
            </div>
          </Card>

          {st.plan === "free" && (
            <Card as="section" aria-labelledby="upgrade-heading" className="flex flex-col gap-4 p-5">
              <div>
                <h2 id="upgrade-heading" className="text-[16px] font-semibold">Upgrade to Pro</h2>
                <p className="mt-1 text-[13px] text-fg-muted">More sessions, natural interviewer voice, longer interviews, full report history.</p>
              </div>
              {st.providers.razorpay || st.providers.stripe ? (
                <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
                  {st.providers.razorpay && (
                    <Button onClick={() => void go("razorpay")} loading={pending === "razorpay"} disabled={pending !== null}>
                      {pending !== "razorpay" && <IndianRupee className="size-4" aria-hidden />} Upgrade (India, INR via Razorpay)
                    </Button>
                  )}
                  {st.providers.stripe && (
                    <Button variant="secondary" onClick={() => void go("stripe")} loading={pending === "stripe"} disabled={pending !== null}>
                      {pending !== "stripe" && <Globe className="size-4" aria-hidden />} Upgrade (international, Stripe)
                    </Button>
                  )}
                </div>
              ) : (
                <Alert tone="neutral">Payments are not enabled on this deployment yet.</Alert>
              )}
            </Card>
          )}

          {msg && <Alert tone="danger">{msg}</Alert>}
          <p className="text-[13px] leading-5 text-fg-muted">Your plan changes once the payment provider confirms the payment. Cancel any time from the provider&apos;s receipt email or by contacting support.</p>
        </div>
      )}
    </div>
  );
}
