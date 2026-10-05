"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

type Status = { plan: string; sessions_this_month: number; sessions_per_month: number;
  subscription: { provider: string; status: string; plan: string } | null; providers: { stripe: boolean; razorpay: boolean } };

export default function Billing() {
  const { user } = useUser();
  const [st, setSt] = useState<Status | null>(null);
  const [msg, setMsg] = useState("");
  useEffect(() => { if (user) void api<Status>("/billing/status").then(setSt); }, [user]);
  if (!user || !st) return null;
  async function go(provider: "stripe" | "razorpay") {
    try {
      const r = await api<{ url: string }>(provider === "stripe" ? "/billing/stripe/checkout" : "/billing/razorpay/subscription",
        { method: "POST", json: { plan: "pro" } });
      window.location.href = r.url;
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Could not start checkout.");
    }
  }
  return (
    <div className="stack" style={{ maxWidth: 720 }}>
      <h1>Plan and billing</h1>
      <div className="card">
        <p>Current plan: <strong>{st.plan}</strong>{st.subscription && <> · {st.subscription.provider} subscription {st.subscription.status}</>}</p>
        <p className="muted">Sessions this month: {st.sessions_this_month} of {st.sessions_per_month}</p>
      </div>
      {st.plan === "free" && (
        <div className="row">
          {st.providers.razorpay && <button className="btn primary" onClick={() => void go("razorpay")}>Upgrade (India, INR via Razorpay)</button>}
          {st.providers.stripe && <button className="btn" onClick={() => void go("stripe")}>Upgrade (international, Stripe)</button>}
          {!st.providers.razorpay && !st.providers.stripe && <p className="muted">Payments are not enabled on this deployment yet.</p>}
        </div>
      )}
      <p className="small muted">Your plan changes once the payment provider confirms the payment. Cancel any time from the provider&apos;s receipt email or by contacting support.</p>
      {msg && <p className="notice error">{msg}</p>}
    </div>
  );
}
