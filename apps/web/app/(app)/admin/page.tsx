"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Alert, Card, cn, Skeleton, Stat } from "@/components/ui";
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

  const unpriced = o ? Object.entries(o.cost.prices_configured).filter(([, v]) => !v).map(([k]) => k) : [];

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-8">
      <header>
        <p className="text-[13px] font-medium text-accent">Admin</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9">Operations{o ? ` (last ${o.days} days)` : ""}</h1>
      </header>

      {!user ? null : err ? (
        <Alert tone="danger">{err}</Alert>
      ) : !o ? (
        <div className="flex flex-col gap-4" aria-busy="true" aria-label="Loading">
          <div className="grid gap-4 sm:grid-cols-3"><Skeleton className="h-24" /><Skeleton className="h-24" /><Skeleton className="h-24" /></div>
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Card className="p-5"><Stat label="Sessions" value={o.sessions} /></Card>
            <Card className="p-5"><Stat label="Cost per session" value={`p50 ${usd(o.cost.session_p50_micro_usd)}`} sub={`p95 ${usd(o.cost.session_p95_micro_usd)}`} /></Card>
            <Card className="p-5"><Stat label="Users" value={o.users} /></Card>
          </div>

          {unpriced.length > 0 && (
            <Alert tone="warn">No price configured for: {unpriced.join(", ")}. Those units are counted but cost $0 until PRICE_TABLE_JSON is set.</Alert>
          )}

          <Section title="Model quality">
            <Table
              head={["Provider / task", "Calls", "Schema-valid (raw)", "Fallback rate", "p95 latency"]}
              numeric={[1, 2, 3, 4]}
              rows={Object.entries(o.llm_quality).map(([k, v]) => ({
                key: k,
                cells: [k, v.calls, `${(v.raw_schema_valid_rate * 100).toFixed(1)}%`, `${(v.fallback_rate * 100).toFixed(1)}%`, `${v.p95_latency_ms} ms`],
              }))}
            />
          </Section>

          <Section title="Latency">
            <Table
              head={["Metric", "n", "p50", "p95"]}
              numeric={[1, 2, 3]}
              rows={Object.entries(o.latency).map(([k, v]) => ({
                key: k,
                cells: [k, v.n, `${Math.round(v.p50_ms)} ms`, `${Math.round(v.p95_ms)} ms`],
              }))}
            />
          </Section>

          <Section title="Usage by unit">
            <Table
              head={["Unit | provider", "Quantity", "Cost"]}
              numeric={[1, 2]}
              rows={Object.entries(o.cost.by_kind_provider).map(([k, v]) => ({
                key: k,
                cells: [k, Math.round(v.quantity), usd(v.cost_micro_usd)],
              }))}
            />
          </Section>

          <Section title="Most expensive sessions">
            <Table
              head={["Session", "User", "Plan", "Total", "Breakdown"]}
              numeric={[3]}
              rows={o.cost.per_session.slice(0, 50).map((r) => ({
                key: r.session_id,
                cells: [
                  <span key="id" className="font-mono">{r.session_id.slice(0, 8)}</span>,
                  r.user,
                  r.plan,
                  usd(r.total_micro_usd),
                  <span key="b" className="text-[12px] text-fg-muted">{Object.entries(r.by_kind).map(([k, v]) => `${k}: ${usd(v)}`).join(", ")}</span>,
                ],
              }))}
            />
          </Section>

          <Section title="Deployment">
            <Card className="overflow-x-auto p-4">
              <pre className="font-mono text-[12px] leading-5 text-fg-muted">{JSON.stringify({ capabilities: o.capabilities, flags: o.flags }, null, 2)}</pre>
            </Card>
          </Section>
        </>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-[18px] font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function Table({ head, rows, numeric = [] }: { head: string[]; rows: { key: string; cells: ReactNode[] }[]; numeric?: number[] }) {
  if (rows.length === 0) return <p className="rounded-[16px] border border-dashed border-line px-4 py-6 text-center text-[13px] text-fg-muted">No data in this period.</p>;
  return (
    <Card className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-left text-[13px]">
        <thead>
          <tr className="border-b border-line bg-surface-2">
            {head.map((h, i) => (
              <th key={h} scope="col" className={cn("px-4 py-2.5 text-[12px] font-medium uppercase tracking-[0.06em] text-fg-subtle", numeric.includes(i) && "text-right")}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r.key} className="transition-colors hover:bg-surface-2">
              {r.cells.map((c, i) => (
                <td key={i} className={cn("px-4 py-2.5 align-top text-fg", numeric.includes(i) && "text-right font-mono tabular", i === 0 && "break-all")}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
