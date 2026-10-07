"use client";

import { useEffect, useState } from "react";
import { Button, Skeleton, Switch } from "@/components/ui";
import { api } from "@/lib/api";

type Consents = { version: string; text: Record<string, string>; granted: Record<string, boolean> };
const LABELS: Record<string, { title: string; required?: boolean }> = {
  data_processing: { title: "Run practice interviews", required: true },
  biometric_face: { title: "Face check" },
  biometric_voice: { title: "Voice check" },
  store_recordings: { title: "Keep recordings" },
  model_training: { title: "Help improve the models" },
};

export function ConsentForm({ onDone }: { onDone?: () => void }) {
  const [c, setC] = useState<Consents | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  useEffect(() => { void api<Consents>("/consent").then(setC); }, []);
  if (!c) return <div className="flex flex-col gap-3">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>;

  async function toggle(kind: string, granted: boolean) {
    setSaving(kind);
    // Optimistic: reflect the choice immediately, then reconcile with the server's record.
    setC((prev) => (prev ? { ...prev, granted: { ...prev.granted, [kind]: granted } } : prev));
    try {
      const r = await api<{ granted: Record<string, boolean> }>("/consent", { method: "POST", json: { kind, granted } });
      setC((prev) => (prev ? { ...prev, granted: r.granted } : prev));
    } catch {
      setC((prev) => (prev ? { ...prev, granted: { ...prev.granted, [kind]: !granted } } : prev));
    } finally {
      setSaving(null);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {Object.entries(LABELS).map(([k, l]) => (
        <div key={k} className="flex items-start gap-4 rounded-[16px] border border-line bg-surface p-4" aria-busy={saving === k}>
          <div className="flex-1">
            <p className="text-[14px] font-medium">{l.title} <span className="text-[12px] font-normal text-fg-subtle">{l.required ? "required" : "optional"}</span></p>
            <p id={`consent-${k}`} className="mt-1 text-[13px] leading-5 text-fg-muted">{c.text[k]}</p>
          </div>
          <Switch checked={!!c.granted[k]} label={`${l.title}${l.required ? " (required)" : " (optional)"}`} onChange={(v) => void toggle(k, v)} />
        </div>
      ))}
      <p className="text-[12px] leading-5 text-fg-subtle">Consent version {c.version}. You can change these at any time in Privacy settings; withdrawing a biometric consent deletes that template immediately.</p>
      {onDone && <Button size="lg" disabled={!c.granted.data_processing} onClick={onDone}>Continue</Button>}
    </div>
  );
}
