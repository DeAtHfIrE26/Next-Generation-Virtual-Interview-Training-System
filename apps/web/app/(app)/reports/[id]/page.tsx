"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ReportView, type Report } from "@/components/ReportView";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

export default function ReportPage() {
  const { id } = useParams<{ id: string }>();
  const { user } = useUser();
  const [r, setR] = useState<Report | null>(null);
  const [err, setErr] = useState("");
  const [share, setShare] = useState<string | null>(null);
  useEffect(() => {
    if (user) api<Report>(`/reports/${id}`).then(setR).catch((e) => setErr(e instanceof ApiError ? e.message : "Could not load report"));
  }, [user, id]);
  if (!user) return null;
  if (err) return <p className="notice error">{err}</p>;
  if (!r) return <p className="muted">Loading…</p>;
  return (
    <div className="stack">
      <div className="row">
        <button className="btn" onClick={() => window.print()}>Download PDF</button>
        <button className="btn" onClick={async () => setShare((await api<{ url: string }>(`/reports/${id}/share`, { method: "POST", json: { days: 14 } })).url)}>Create share link</button>
        {share && <button className="btn danger" onClick={async () => { await api(`/reports/${id}/share`, { method: "DELETE" }); setShare(null); }}>Revoke link</button>}
      </div>
      {share && <p className="notice ok">Anyone with this link can view the feedback (not your transcript) for 14 days: <code>{share}</code></p>}
      <ReportView r={r} />
    </div>
  );
}
