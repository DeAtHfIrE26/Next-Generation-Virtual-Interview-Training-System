"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ReportView, type Report } from "@/components/ReportView";
import { api } from "@/lib/api";

export default function Shared() {
  const { token } = useParams<{ token: string }>();
  const [r, setR] = useState<Report | null>(null);
  const [missing, setMissing] = useState(false);
  useEffect(() => { api<Report>(`/shared/${token}`).then(setR).catch(() => setMissing(true)); }, [token]);
  if (missing) return <p className="notice error">This link has expired or was revoked.</p>;
  return r ? <ReportView r={r} /> : <p className="muted">Loading…</p>;
}
