"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

type Overview = {
  days: number; sessions: number; users: number;
  cost: { per_session: { session_id: string; created_at: string; user: string; plan: string; total_micro_usd: number; by_kind: Record<string, number> }[];
    by_kind_provider: Record<string, { quantity: number; cost_micro_usd: number }>;
    session_p50_micro_usd: number | null; session_p95_micro_usd: number | null; prices_configured: Record<string, boolean> };
  llm_quality: Record<string, { calls: number; raw_schema_valid_rate: number; fallback_rate: number; p95_latency_ms: number }>;
  latency: Record<string, { n: number; p50_ms: number; p95_ms: number }>;
  capabilities: Record<string, unknown>; flags: Record<string, boolean>;
};
const usd = (micro: number | null) => (micro === null ? "–" : `$${(micro / 1e6).toFixed(4)}`);

export default function Admin() {
  const { user } = useUser();
  const [o, setO] = useState<Overview | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    if (user) api<Overview>("/admin/overview?days=30").then(setO).catch((e) => setErr(e instanceof ApiError ? e.message : "error"));
  }, [user]);
  if (!user) return null;
  if (err) return <p className="notice error">{err}</p>;
  if (!o) return <p className="muted">Loading…</p>;
  const unpriced = Object.entries(o.cost.prices_configured).filter(([, v]) => !v).map(([k]) => k);
  return (
    <div className="stack">
      <h1>Operations (last {o.days} days)</h1>
      <div className="grid three">
        <div className="card"><h3>Sessions</h3><p style={{ fontSize: 28, margin: 0 }}>{o.sessions}</p></div>
        <div className="card"><h3>Cost per session</h3><p style={{ margin: 0 }}>p50 {usd(o.cost.session_p50_micro_usd)} · p95 {usd(o.cost.session_p95_micro_usd)}</p></div>
        <div className="card"><h3>Users</h3><p style={{ fontSize: 28, margin: 0 }}>{o.users}</p></div>
      </div>
      {unpriced.length > 0 && <p className="notice">No price configured for: {unpriced.join(", ")}. Those units are counted but cost $0 until PRICE_TABLE_JSON is set.</p>}
      <h2>Model quality</h2>
      <table className="data"><thead><tr><th>Provider / task</th><th>Calls</th><th>Schema-valid (raw)</th><th>Fallback rate</th><th>p95 latency</th></tr></thead>
        <tbody>{Object.entries(o.llm_quality).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v.calls}</td><td>{(v.raw_schema_valid_rate * 100).toFixed(1)}%</td><td>{(v.fallback_rate * 100).toFixed(1)}%</td><td>{v.p95_latency_ms} ms</td></tr>)}</tbody></table>
      <h2>Latency</h2>
      <table className="data"><thead><tr><th>Metric</th><th>n</th><th>p50</th><th>p95</th></tr></thead>
        <tbody>{Object.entries(o.latency).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v.n}</td><td>{Math.round(v.p50_ms)} ms</td><td>{Math.round(v.p95_ms)} ms</td></tr>)}</tbody></table>
      <h2>Usage by unit</h2>
      <table className="data"><thead><tr><th>Unit | provider</th><th>Quantity</th><th>Cost</th></tr></thead>
        <tbody>{Object.entries(o.cost.by_kind_provider).map(([k, v]) => <tr key={k}><td>{k}</td><td>{Math.round(v.quantity)}</td><td>{usd(v.cost_micro_usd)}</td></tr>)}</tbody></table>
      <h2>Most expensive sessions</h2>
      <table className="data"><thead><tr><th>Session</th><th>User</th><th>Plan</th><th>Total</th><th>Breakdown</th></tr></thead>
        <tbody>{o.cost.per_session.slice(0, 50).map((r) => <tr key={r.session_id}><td>{r.session_id.slice(0, 8)}</td><td>{r.user}</td><td>{r.plan}</td><td>{usd(r.total_micro_usd)}</td>
          <td className="small">{Object.entries(r.by_kind).map(([k, v]) => `${k}: ${usd(v)}`).join(", ")}</td></tr>)}</tbody></table>
      <h2>Deployment</h2>
      <pre className="small">{JSON.stringify({ capabilities: o.capabilities, flags: o.flags }, null, 2)}</pre>
    </div>
  );
}
