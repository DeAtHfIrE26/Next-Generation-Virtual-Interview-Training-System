"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ReportView, type Report } from "@/components/report/ReportView";
import { EmptyState, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";

export default function Shared() {
  const { token } = useParams<{ token: string }>();
  const [r, setR] = useState<Report | null>(null);
  const [missing, setMissing] = useState(false);
  useEffect(() => { api<Report>(`/shared/${token}`).then(setR).catch(() => setMissing(true)); }, [token]);
  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-10">
      {missing && <EmptyState title="This link has expired or was revoked" body="Ask the person who shared it for a new link." />}
      {!r && !missing && <Skeleton className="h-96" />}
      {r && (
        <>
          <header>
            <p className="text-[13px] font-medium text-accent">Shared interview feedback</p>
            <h1 className="mt-1 text-[30px] font-semibold leading-9">{r.role}{r.company ? ` · ${r.company}` : ""}</h1>
            <p className="mt-1 text-[13px] text-fg-muted">Feedback only. The candidate&rsquo;s transcript and integrity details are not shared.</p>
          </header>
          <ReportView r={r} shared />
        </>
      )}
    </div>
  );
}
