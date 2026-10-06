"use client";

import { useState } from "react";
import { Code2, Play, Send } from "lucide-react";
import { Badge, Button, Card, Select, Textarea } from "@/components/ui";
import { api, ApiError } from "@/lib/api";

export type Challenge = {
  id: string; title: string; prompt: string; languages: string[]; starter: Record<string, string>;
  examples: { stdin: string; expected: string }[]; hidden_tests: number;
};
type Outcome = { passed: boolean; hidden: boolean; status: string; stdout: string | null; expected: string | null };

/** Coding challenge attached to the session: sandboxed runs against visible and hidden tests. */
export function CodePanel({ sessionId, challenge }: { sessionId: string; challenge: Challenge }) {
  const first = challenge.languages[0] ?? "python";
  const [lang, setLang] = useState(first);
  const [src, setSrc] = useState(challenge.starter[first] ?? "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ summary: string; error: string | null; outcomes: Outcome[] } | null>(null);
  const [err, setErr] = useState("");

  async function run(final: boolean) {
    setBusy(true);
    setErr("");
    try {
      setResult(await api(`/sessions/${sessionId}/code`, { method: "POST", json: { language: lang, source: src, final } }));
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : "Could not run code.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="flex flex-col gap-3 p-4">
      <div className="flex items-center gap-2">
        <Code2 className="size-4 text-accent" />
        <h3 className="text-[14px] font-semibold">{challenge.title}</h3>
        <Badge className="ml-auto">{challenge.hidden_tests} hidden tests</Badge>
      </div>
      <p className="text-[13px] leading-5 text-fg-muted">{challenge.prompt}</p>
      {challenge.examples.map((e, i) => (
        <pre key={i} className="overflow-x-auto rounded-[10px] bg-surface-2 p-2 font-mono text-[12px]">input: {e.stdin || "(none)"}{"\n"}expected: {e.expected}</pre>
      ))}
      <Select aria-label="Language" value={lang} onChange={(e) => { setLang(e.target.value); setSrc(challenge.starter[e.target.value] ?? ""); }}>
        {challenge.languages.map((l) => <option key={l}>{l}</option>)}
      </Select>
      <Textarea aria-label="Code editor" spellCheck={false} value={src} onChange={(e) => setSrc(e.target.value)} rows={10} className="font-mono text-[12px]" />
      <div className="flex gap-2">
        <Button variant="secondary" size="sm" onClick={() => void run(false)} loading={busy}><Play className="size-3.5" /> Run tests</Button>
        <Button size="sm" onClick={() => void run(true)} disabled={busy}><Send className="size-3.5" /> Submit</Button>
      </div>
      {err && <p className="text-[12px] text-danger">{err}</p>}
      {result && (
        <div className="flex flex-col gap-1 text-[12px]" aria-live="polite">
          <p className="font-medium">{result.summary}</p>
          {result.error && <pre className="whitespace-pre-wrap text-danger">{result.error}</pre>}
          {result.outcomes.map((o, i) => (
            <p key={i} className={o.passed ? "text-live" : "text-warn"}>
              Test {i + 1}{o.hidden ? " (hidden)" : ""}: {o.passed ? "passed" : o.status}
              {!o.hidden && !o.passed && <span className="block font-mono text-fg-muted">expected {o.expected} · got {o.stdout}</span>}
            </p>
          ))}
        </div>
      )}
    </Card>
  );
}
