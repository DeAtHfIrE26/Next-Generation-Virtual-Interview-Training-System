"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowLeft, Copy, Download, Link2, Link2Off, RotateCcw } from "lucide-react";
import { ReportView, type Report } from "@/components/report/ReportView";
import { Alert, Button, ButtonLink, Card, Skeleton } from "@/components/ui";
import { api, ApiError } from "@/lib/api";

export default function ReportPage() {
  const { id } = useParams<{ id: string }>();
  const [r, setR] = useState<Report | null>(null);
  const [err, setErr] = useState("");
  const [share, setShare] = useState<{ url: string; expires_at: string } | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const rep = await api<Report>(`/reports/${id}`).catch(async (e) => {
          // Not built yet (e.g. the tab closed during the interview): finish now, then load.
          if (e instanceof ApiError && e.status === 404) {
            await api(`/sessions/${id}/finish`, { method: "POST" });
            return api<Report>(`/reports/${id}`);
          }
          throw e;
        });
        if (alive) setR(rep);
      } catch (e) {
        if (alive) setErr(e instanceof ApiError ? e.message : "Could not load the report.");
      }
    })();
    return () => { alive = false; };
  }, [id]);

  async function createShare() {
    setShare(await api(`/reports/${id}/share`, { method: "POST", json: { days: 14 } }));
  }

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-8">
      <Link href="/dashboard" className="inline-flex w-fit items-center gap-1 text-[13px] text-fg-muted hover:text-fg"><ArrowLeft className="size-4" /> Dashboard</Link>
      {err && <Alert tone="danger" title="Couldn't load the report">{err}</Alert>}
      {!r && !err && (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-10 w-80" />
          <div className="grid gap-4 lg:grid-cols-[320px_1fr]"><Skeleton className="h-64" /><Skeleton className="h-64" /></div>
          <Skeleton className="h-40" />
        </div>
      )}
      {r && (
        <>
          <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[13px] font-medium text-accent">Interview report</p>
              <h1 className="mt-1 text-[30px] font-semibold leading-9">{r.role}{r.company ? ` · ${r.company}` : ""}</h1>
              <p className="mt-1 text-[14px] text-fg-muted">
                {[r.interview_type, r.round, r.seniority].filter(Boolean).map((x) => String(x).replace(/_/g, " ")).join(" · ")} · {new Date(r.generated_at).toLocaleString()}
              </p>
            </div>
            <div className="flex flex-wrap gap-2 print:hidden">
              <Button variant="secondary" onClick={() => window.print()}><Download className="size-4" /> PDF</Button>
              {share ? (
                <Button variant="secondary" onClick={async () => { await api(`/reports/${id}/share`, { method: "DELETE" }); setShare(null); }}><Link2Off className="size-4" /> Revoke link</Button>
              ) : (
                <Button variant="secondary" onClick={() => void createShare()}><Link2 className="size-4" /> Share</Button>
              )}
              <ButtonLink href="/interview/new"><RotateCcw className="size-4" /> Practise again</ButtonLink>
            </div>
          </header>
          {share && (
            <Card className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center">
              <p className="flex-1 text-[13px] text-fg-muted">Anyone with this link can see the feedback (not your transcript) until {new Date(share.expires_at).toLocaleDateString()}.</p>
              <code className="truncate rounded-[8px] bg-surface-2 px-2 py-1 font-mono text-[12px]" data-testid="share-url">{share.url}</code>
              <Button size="sm" variant="secondary" onClick={() => { void navigator.clipboard?.writeText(share.url); setCopied(true); }}><Copy className="size-3.5" /> {copied ? "Copied" : "Copy"}</Button>
            </Card>
          )}
          <ReportView r={r} />
        </>
      )}
    </div>
  );
}
