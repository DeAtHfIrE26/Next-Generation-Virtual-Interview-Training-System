"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ScoreTrend } from "@/components/Charts";
import { api } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

type Row = { id: string; role: string; seniority: string; mode: string; status: string; created_at: string;
  overall: number | null; label: string | null };

export default function Dashboard() {
  const { user } = useUser();
  const [rows, setRows] = useState<Row[] | null>(null);
  useEffect(() => { if (user) void api<Row[]>("/sessions").then(setRows); }, [user]);
  if (!user) return null;
  const finished = (rows ?? []).filter((r) => r.overall !== null).reverse();
  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Hi{user.name ? `, ${user.name}` : ""}</h1>
        <Link className="btn primary" href="/interview/new">New practice interview</Link>
      </div>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Overall score by session</h2><span className="badge">experimental</span>
        </div>
        <ScoreTrend title="Overall score by session" points={finished.map((r) => ({
          label: new Date(r.created_at).toLocaleDateString(), value: r.overall, note: r.role }))} />
      </section>
      <section>
        <h2>Sessions</h2>
        {rows === null ? <p className="muted">Loading…</p> : rows.length === 0 ? (
          <p className="muted">No sessions yet. <Link href="/interview/new">Start your first one.</Link></p>
        ) : (
          <table className="data">
            <thead><tr><th>Date</th><th>Role</th><th>Status</th><th>Overall</th><th /></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.id}>
                <td>{new Date(r.created_at).toLocaleString()}</td>
                <td>{r.role} <span className="muted small">({r.seniority}{r.mode === "proctored" ? ", strict" : ""})</span></td>
                <td>{r.status.replace(/_/g, " ")}</td>
                <td>{r.overall === null ? "–" : `${Math.round(r.overall * 100)}%`}</td>
                <td>{r.status === "active" ? <Link href={`/interview/${r.id}`}>Resume</Link> : <Link href={`/reports/${r.id}`}>Report</Link>}</td>
              </tr>))}</tbody>
          </table>
        )}
      </section>
      <p className="small muted"><Link href="/enroll">Set up optional face and voice checks</Link></p>
    </div>
  );
}
