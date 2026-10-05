"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Consents = { version: string; text: Record<string, string>; granted: Record<string, boolean> };
const LABELS: Record<string, string> = {
  data_processing: "Run practice interviews (required)",
  biometric_face: "Face check (optional)",
  biometric_voice: "Voice check (optional)",
  store_recordings: "Keep recordings (optional)",
  model_training: "Help improve the models (optional)",
};

export function ConsentForm({ onDone }: { onDone?: () => void }) {
  const [c, setC] = useState<Consents | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  useEffect(() => { void api<Consents>("/consent").then(setC); }, []);
  if (!c) return <p className="muted">Loading…</p>;
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
    <div className="stack">
      {Object.keys(LABELS).map((k) => (
        <label key={k} className="card check" style={{ alignItems: "flex-start" }}>
          <input type="checkbox" checked={!!c.granted[k]} aria-busy={saving === k}
            onChange={(e) => void toggle(k, e.target.checked)} aria-describedby={`consent-${k}`} />
          <span><strong>{LABELS[k]}</strong><br /><span id={`consent-${k}`} className="muted small">{c.text[k]}</span></span>
        </label>
      ))}
      <p className="small muted">Consent version {c.version}. You can change these at any time in Privacy settings; withdrawing a biometric consent deletes that template immediately.</p>
      {onDone && <button className="btn primary" disabled={!c.granted.data_processing} onClick={onDone}>Continue</button>}
    </div>
  );
}
