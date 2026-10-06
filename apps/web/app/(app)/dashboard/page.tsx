"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, BarChart3, CalendarClock, Fingerprint, Plus, Trophy } from "lucide-react";
import { Badge, ButtonLink, Card, EmptyState, Skeleton, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

type Dim = "relevance" | "structure" | "depth" | "communication" | "technical_accuracy";
type Row = {
  id: string; role: string; company: string | null; interview_type: string | null; seniority: string; mode: string; status: string;
  created_at: string; overall: number | null; label: string | null; dimensions: Partial<Record<Dim, number | null>> | null;
};
const DIMS: [Dim, string][] = [["relevance", "Relevance"], ["structure", "Structure"], ["depth", "Depth"], ["communication", "Communication"], ["technical_accuracy", "Technical"]];

export default function Dashboard() {
  const { user } = useUser();
  const [rows, setRows] = useState<Row[] | null>(null);
  useEffect(() => { if (user) void api<Row[]>("/sessions").then(setRows).catch(() => setRows([])); }, [user]);
  const scored = (rows ?? []).filter((r) => r.overall !== null).slice().reverse(); // oldest first
  const avg = scored.length ? scored.reduce((s, r) => s + (r.overall ?? 0), 0) / scored.length : null;
  const best = scored.length ? Math.max(...scored.map((r) => r.overall ?? 0)) : null;
  const active = (rows ?? []).find((r) => r.status === "active");

  return (
    <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-5 py-8">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[13px] font-medium text-accent">Dashboard</p>
          <h1 className="mt-1 text-[30px] font-semibold leading-9">{user ? `Hi${user.name ? `, ${user.name.split(" ")[0]}` : ""}` : <Skeleton className="h-9 w-40" />}</h1>
        </div>
        <ButtonLink href="/interview/new" size="lg"><Plus className="size-4" /> New interview</ButtonLink>
      </header>

      {active && (
        <Card className="flex flex-col gap-3 border-accent/40 p-4 sm:flex-row sm:items-center">
          <CalendarClock className="size-5 text-accent" />
          <p className="flex-1 text-[14px]">You have an interview in progress: <span className="font-medium">{active.role}</span>.</p>
          <ButtonLink href={`/interview/${active.id}`} variant="secondary" size="sm">Resume <ArrowRight className="size-3.5" /></ButtonLink>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Card className="p-5"><Stat label="Interviews" value={rows ? rows.length : "–"} sub="all time" /></Card>
        <Card className="p-5"><Stat label="Average score" value={avg !== null ? Math.round(avg * 100) : "–"} sub="out of 100" /></Card>
        <Card className="p-5"><Stat label="Best score" value={best !== null ? Math.round(best * 100) : "–"} sub={<span className="inline-flex items-center gap-1"><Trophy className="size-3" /> personal best</span>} /></Card>
        <Card className="p-5"><Stat label="Last change" value={scored.length > 1 ? delta(scored.at(-1)!.overall!, scored.at(-2)!.overall!) : "–"} sub="vs previous interview" /></Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <Card className="flex flex-col gap-3 p-5">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-[16px] font-semibold"><BarChart3 className="size-4 text-accent" /> Progress</h2>
            <Badge>experimental</Badge>
          </div>
          {rows === null ? <Skeleton className="h-48" /> : scored.length < 2 ? (
            <p className="py-10 text-center text-[13px] text-fg-muted">Complete two interviews to see your progress over time.</p>
          ) : <Progress points={scored.map((r) => ({ v: r.overall!, label: new Date(r.created_at).toLocaleDateString(), note: r.role }))} />}
        </Card>
        <Card className="flex flex-col gap-3 p-5">
          <h2 className="text-[16px] font-semibold">Skills: latest vs first</h2>
          {scored.length === 0 ? <p className="py-10 text-center text-[13px] text-fg-muted">Your skill breakdown appears after your first interview.</p> : (
            <ul className="flex flex-col gap-3">
              {DIMS.map(([k, label]) => {
                const first = scored[0]?.dimensions?.[k] ?? null, last = scored.at(-1)?.dimensions?.[k] ?? null;
                return (
                  <li key={k} className="flex items-center gap-3 text-[13px]">
                    <span className="w-28 text-fg-muted">{label}</span>
                    <div className="relative h-2 flex-1 rounded-full bg-surface-3">
                      {first !== null && <span className="absolute top-1/2 size-2.5 -translate-y-1/2 rounded-full border-2 border-fg-subtle bg-canvas" style={{ left: `calc(${((first - 1) / 4) * 100}% - 5px)` }} title={`first ${first.toFixed(1)}`} />}
                      {last !== null && <span className="absolute top-1/2 size-2.5 -translate-y-1/2 rounded-full bg-accent" style={{ left: `calc(${((last - 1) / 4) * 100}% - 5px)` }} title={`latest ${last.toFixed(1)}`} />}
                    </div>
                    <span className="w-8 text-right font-mono tabular">{last === null ? "–" : last.toFixed(1)}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      </div>

      <section>
        <h2 className="mb-3 text-[18px] font-semibold">Interviews</h2>
        {rows === null ? <Skeleton className="h-40" /> : rows.length === 0 ? (
          <EmptyState title="No interviews yet" body="Your first practice interview takes about two minutes to set up." action={<ButtonLink href="/interview/new">Start practising</ButtonLink>} />
        ) : (
          <Card className="divide-y divide-line">
            {rows.map((r) => (
              <Link key={r.id} href={r.status === "active" ? `/interview/${r.id}` : `/reports/${r.id}`} className="flex items-center gap-4 px-4 py-3 transition-colors hover:bg-surface-2">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[14px] font-medium">{r.role}{r.company ? ` · ${r.company}` : ""}</p>
                  <p className="text-[12px] text-fg-subtle">{new Date(r.created_at).toLocaleString()} · {(r.interview_type ?? "").replace(/_/g, " ")} · {r.seniority}{r.mode === "proctored" ? " · strict" : ""}</p>
                </div>
                {r.status === "active" ? <Badge tone="accent">in progress</Badge> : r.status !== "completed" ? <Badge>{r.status.replace(/_/g, " ")}</Badge> : null}
                <span className="w-12 text-right font-mono text-[16px] font-semibold tabular">{r.overall === null ? "–" : Math.round(r.overall * 100)}</span>
                <ArrowRight className="size-4 text-fg-subtle" />
              </Link>
            ))}
          </Card>
        )}
      </section>

      <Link href="/enroll" className="inline-flex w-fit items-center gap-2 text-[13px] text-fg-muted hover:text-fg"><Fingerprint className="size-4" /> Optional: set up face and voice checks</Link>
    </div>
  );
}

function delta(a: number, b: number) {
  const d = Math.round((a - b) * 100);
  return d > 0 ? `+${d}` : `${d}`;
}

function Progress({ points }: { points: { v: number; label: string; note: string }[] }) {
  const W = 640, H = 200, P = 28, n = points.length;
  const x = (i: number) => P + (i * (W - 2 * P)) / Math.max(1, n - 1);
  const y = (v: number) => H - P - v * (H - 2 * P);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.v)}`).join(" ");
  const area = `${line} L${x(n - 1)},${H - P} L${x(0)},${H - P} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-52 w-full" role="img" aria-label={`Overall score by interview: ${points.map((p) => Math.round(p.v * 100)).join(", ")}`}>
      <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--accent)" stopOpacity="0.25" /><stop offset="1" stopColor="var(--accent)" stopOpacity="0" /></linearGradient></defs>
      {[0, 0.5, 1].map((g) => <g key={g}><line x1={P} x2={W - P} y1={y(g)} y2={y(g)} stroke="var(--line)" strokeDasharray="3 5" /><text x={4} y={y(g) + 4} fontSize="10" fill="var(--fg-subtle)">{g * 100}</text></g>)}
      <path d={area} fill="url(#g)" />
      <path d={line} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinejoin="round" />
      {points.map((p, i) => <circle key={i} cx={x(i)} cy={y(p.v)} r="4" fill="var(--canvas)" stroke="var(--accent)" strokeWidth="2"><title>{`${p.label} · ${p.note}: ${Math.round(p.v * 100)}`}</title></circle>)}
    </svg>
  );
}
