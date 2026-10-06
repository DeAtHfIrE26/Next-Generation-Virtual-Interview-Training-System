"use client";

import { useState } from "react";
import { AlertTriangle, ChevronDown, Eye, Gauge, Lightbulb, MessageSquareQuote, Mic2, ShieldCheck, Sparkles, Target, TrendingUp } from "lucide-react";
import { Badge, Card, cn, Stat } from "@/components/ui";

// ----------------------------------------------------------------------------- types (report v2)

type Dim = "relevance" | "structure" | "depth" | "communication" | "technical_accuracy";
type Scores = Partial<Record<Dim, number | null>>;
interface Measured { measured: boolean; [k: string]: unknown }
export interface ReportAnswer {
  index: number; question: string; action: string; competency: string; difficulty: number; emergency_question: boolean;
  anchor_quote: string; answer: string; answer_seconds: number; interviewer_read: number | null; scores: Scores | null;
  overall: number | null; label: string; method: string; star?: Record<string, boolean> | null;
  evidence: { quote: string; supports?: string; dimension?: string }[]; strengths: string[]; improvements: string[];
  delivery: Measured; gaze: Measured; lipsync?: Measured; voice?: Measured;
}
export interface Report {
  version: number; generated_at: string; session_id: string; role: string; seniority?: string; company: string | null;
  interview_type?: string; round?: string; mode?: string;
  summary: {
    answers: number; overall: number | null; label: string; dimensions: Scores; difficulty_trajectory: number[]; final_difficulty?: number;
    words_per_minute: number | null; filler_per_100_words: number | null; off_screen_fraction: number | null; emergency_questions: number; duration_minutes?: number;
  };
  blueprint?: { summary: string; emergency: boolean };
  skills: { id: string; name: string; why: string; questions: number; overall: number | null; interviewer_read: number | null; covered: boolean; dimensions: Scores }[];
  moments: { kind: "strongest" | "needs_work"; index: number; question: string; quote: string | null; overall: number; why: string[] }[];
  tips: { tip: string; from_answer: number }[];
  observations: string[];
  answers: ReportAnswer[];
  integrity?: { mode: string; events: Record<string, number>; note: string };
  code?: { challenge_id?: string; summary?: string; passed?: number; total?: number }[];
  transcript?: { speaker: "interviewer" | "candidate"; text: string; index: number }[];
  appendix?: { prototype_nine_factor: { score: number; breakdown: Record<string, number>; label: string } };
}

const DIM_LABEL: Record<Dim, string> = {
  relevance: "Relevance", structure: "Structure", depth: "Depth", communication: "Communication", technical_accuracy: "Technical accuracy",
};

const pct = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}`);
const five = (x: number | null | undefined) => (x === null || x === undefined ? null : x);

// ----------------------------------------------------------------------------- view

export function ReportView({ r, shared }: { r: Report; shared?: boolean }) {
  const s = r.summary;
  return (
    <div className="flex flex-col gap-6">
      {s.emergency_questions > 0 && (
        <Card className="flex items-start gap-3 border-warn/40 p-4">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warn" />
          <p className="text-[14px] text-fg-muted">
            {s.emergency_questions} question{s.emergency_questions > 1 ? "s were" : " was"} a backup question because the AI interviewer was unavailable at the time.
            Those answers are scored the same way but the questions were not tailored to you.
          </p>
        </Card>
      )}

      {/* Overview */}
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card className="flex flex-col items-center justify-center gap-3 p-6 text-center">
          <ScoreRing value={s.overall} />
          <div className="flex items-center gap-2">
            <Badge tone={s.label === "calibrated" ? "success" : "neutral"}>{s.label}</Badge>
            <span className="text-[12px] text-fg-subtle">{s.answers} answer{s.answers === 1 ? "" : "s"} scored</span>
          </div>
          <p className="max-w-[240px] text-[12px] leading-5 text-fg-subtle">
            Average of the five rubric dimensions. Every score cites your own words below. &ldquo;Experimental&rdquo; until validated against human raters.
          </p>
        </Card>
        <Card className="flex flex-col gap-5 p-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Speaking pace" value={s.words_per_minute ? `${Math.round(s.words_per_minute)}` : "–"} sub="words per minute" />
            <Stat label="Filler words" value={s.filler_per_100_words !== null ? s.filler_per_100_words.toFixed(1) : "–"} sub="per 100 words" />
            <Stat label="Eye contact" value={s.off_screen_fraction !== null ? `${Math.round((1 - s.off_screen_fraction) * 100)}%` : "–"} sub={s.off_screen_fraction !== null ? "of answer time" : "camera off"} />
            <Stat label="Final difficulty" value={s.final_difficulty ? `${s.final_difficulty}/5` : "–"} sub="adapted to your answers" />
          </div>
          <div className="grid gap-5 sm:grid-cols-[1fr_220px]">
            <div className="flex flex-col gap-2.5">
              {(Object.keys(DIM_LABEL) as Dim[]).map((d) => <DimBar key={d} label={DIM_LABEL[d]} value={five(s.dimensions[d])} />)}
            </div>
            <div>
              <p className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-fg-muted"><TrendingUp className="size-3.5" /> Difficulty by question</p>
              <Trajectory values={s.difficulty_trajectory} />
            </div>
          </div>
        </Card>
      </div>

      {/* Skills */}
      {r.skills.length > 0 && (
        <section>
          <SectionTitle icon={<Target className="size-4" />} title="By skill" sub={r.blueprint?.summary} />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {r.skills.map((k) => (
              <Card key={k.id} className={cn("flex flex-col gap-2 p-4", !k.covered && "border-dashed bg-transparent")}>
                <div className="flex items-start justify-between gap-2">
                  <h3 className="text-[15px] font-semibold">{k.name}</h3>
                  <span className="font-mono text-[18px] font-semibold tabular">{k.covered ? pct(k.overall) : "–"}</span>
                </div>
                <p className="line-clamp-2 text-[12px] leading-5 text-fg-muted">{k.why}</p>
                <p className="mt-auto text-[12px] text-fg-subtle">{k.covered ? `${k.questions} question${k.questions > 1 ? "s" : ""}` : "Not reached in this session"}</p>
              </Card>
            ))}
          </div>
        </section>
      )}

      {/* Moments + tips */}
      <div className="grid gap-4 lg:grid-cols-2">
        {r.moments.length > 0 && (
          <section>
            <SectionTitle icon={<MessageSquareQuote className="size-4" />} title="Your moments" />
            <div className="flex flex-col gap-3">
              {r.moments.map((m) => (
                <Card key={m.kind} className="flex flex-col gap-2 p-4">
                  <div className="flex items-center gap-2">
                    <Badge tone={m.kind === "strongest" ? "success" : "warn"}>{m.kind === "strongest" ? "Strongest answer" : "Most room to grow"}</Badge>
                    <span className="ml-auto font-mono text-[13px] tabular text-fg-muted">{pct(m.overall)}</span>
                  </div>
                  <p className="text-[13px] text-fg-muted">Q{m.index + 1}: {m.question}</p>
                  {m.quote && <blockquote className="border-l-2 border-accent pl-3 text-[14px] italic leading-6">&ldquo;{m.quote}&rdquo;</blockquote>}
                  {m.why.length > 0 && <ul className="list-disc pl-5 text-[13px] leading-5 text-fg-muted">{m.why.map((w, i) => <li key={i}>{w}</li>)}</ul>}
                </Card>
              ))}
            </div>
          </section>
        )}
        {(r.tips.length > 0 || r.observations.length > 0) && (
          <section>
            <SectionTitle icon={<Lightbulb className="size-4" />} title="What to practise next" />
            <Card className="flex flex-col gap-3 p-4">
              <ol className="flex flex-col gap-3">
                {r.tips.map((t, i) => (
                  <li key={i} className="flex gap-3 text-[14px] leading-6">
                    <span className="grid size-6 shrink-0 place-items-center rounded-full bg-accent/15 font-mono text-[12px] text-accent">{i + 1}</span>
                    <span>{t.tip} <a href={`#answer-${t.from_answer}`} className="text-[12px] text-fg-subtle hover:text-accent">(Q{t.from_answer + 1})</a></span>
                  </li>
                ))}
              </ol>
              {r.observations.length > 0 && (
                <div className="border-t border-line pt-3">
                  <p className="mb-1 text-[12px] font-medium text-fg-muted">Delivery observations</p>
                  <ul className="list-disc pl-5 text-[13px] leading-5 text-fg-muted">{r.observations.map((o, i) => <li key={i}>{o}</li>)}</ul>
                </div>
              )}
            </Card>
          </section>
        )}
      </div>

      {/* Answers */}
      <section>
        <SectionTitle icon={<Sparkles className="size-4" />} title="Answer by answer" />
        <div className="flex flex-col gap-3">
          {r.answers.map((a) => <AnswerCard key={a.index} a={a} shared={shared} />)}
          {r.answers.length === 0 && <Card className="p-6 text-[14px] text-fg-muted">No answers were recorded in this session.</Card>}
        </div>
      </section>

      {!shared && r.integrity && (
        <section>
          <SectionTitle icon={<ShieldCheck className="size-4" />} title="Integrity checks" />
          <Card className="p-4 text-[14px]">
            {Object.keys(r.integrity.events).length === 0 ? (
              <p className="text-fg-muted">No integrity notices in this session.</p>
            ) : (
              <ul className="flex flex-wrap gap-2">{Object.entries(r.integrity.events).map(([k, v]) => <Badge key={k}>{k.replace(/_/g, " ")}: {v}</Badge>)}</ul>
            )}
            <p className="mt-2 text-[12px] text-fg-subtle">{r.integrity.note}</p>
          </Card>
        </section>
      )}

      {!shared && r.transcript && r.transcript.length > 0 && (
        <Collapsible title="Full transcript">
          <ol className="flex flex-col gap-2 text-[14px] leading-6">
            {r.transcript.map((l, i) => (
              <li key={i}><span className="font-medium">{l.speaker === "interviewer" ? "Interviewer" : "You"}:</span> <span className="text-fg-muted">{l.text}</span></li>
            ))}
          </ol>
        </Collapsible>
      )}

      {!shared && r.appendix && (
        <Collapsible title="Appendix: original prototype score">
          <p className="text-[13px] text-fg-muted">
            The research prototype&rsquo;s nine-factor score, kept so the original analysis stays reproducible: <span className="font-mono text-fg">{r.appendix.prototype_nine_factor.score}</span> ({r.appendix.prototype_nine_factor.label}).
          </p>
        </Collapsible>
      )}
    </div>
  );
}

function AnswerCard({ a, shared }: { a: ReportAnswer; shared?: boolean }) {
  const [open, setOpen] = useState(false);
  const d = a.delivery;
  return (
    <Card id={`answer-${a.index}`} className="overflow-hidden">
      <button type="button" className="flex w-full items-start gap-3 p-4 text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="mt-0.5 font-mono text-[12px] text-fg-subtle">Q{a.index + 1}</span>
        <span className="flex-1">
          <span className="block text-[14px] font-medium leading-6">{a.question}</span>
          <span className="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-fg-subtle">
            <span>{a.competency}</span>·<span>difficulty {a.difficulty}/5</span>
            {a.action === "follow_up" && <Badge>follow-up</Badge>}
            {a.action === "challenge" && <Badge>challenge</Badge>}
            {a.emergency_question && <Badge tone="warn">backup question</Badge>}
            {a.method !== "llm" && <Badge>{a.method === "heuristic" ? "offline scoring" : a.method}</Badge>}
          </span>
        </span>
        <span className="font-mono text-[18px] font-semibold tabular">{pct(a.overall)}</span>
        <ChevronDown className={cn("mt-1 size-4 text-fg-subtle transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="flex flex-col gap-4 border-t border-line p-4">
          {!shared && <p className="rounded-[12px] bg-surface-2 p-3 text-[14px] leading-6"><Highlight text={a.answer} quotes={a.evidence.map((e) => e.quote)} /></p>}
          {a.scores && (
            <div className="grid gap-2 sm:grid-cols-2">
              {(Object.keys(DIM_LABEL) as Dim[]).map((k) => <DimBar key={k} label={DIM_LABEL[k]} value={five(a.scores?.[k])} compact />)}
            </div>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            {a.strengths.length > 0 && <List title="What worked" items={a.strengths} tone="text-live" />}
            {a.improvements.length > 0 && <List title="To improve" items={a.improvements} tone="text-warn" />}
          </div>
          {a.evidence.length > 0 && (
            <div>
              <p className="mb-1 text-[12px] font-medium text-fg-muted">Evidence (quoted from your answer, checked against the transcript)</p>
              <ul className="flex flex-col gap-1 text-[13px]">
                {a.evidence.map((e, i) => <li key={i}><span className="italic">&ldquo;{e.quote}&rdquo;</span>{e.supports ? <span className="text-fg-subtle"> — {e.supports}</span> : null}</li>)}
              </ul>
            </div>
          )}
          <div className="flex flex-wrap gap-2 text-[12px]">
            <Chip icon={<Gauge className="size-3" />} on={d.measured} text={d.measured ? `${Math.round(Number(d.words_per_minute ?? 0))} wpm · ${Number(d.filler_per_100_words ?? 0).toFixed(1)} fillers/100 · longest pause ${Number(d.longest_pause_s ?? 0).toFixed(1)}s` : "delivery not measured (typed)"} />
            <Chip icon={<Eye className="size-3" />} on={a.gaze.measured} text={a.gaze.measured ? `eye contact ${Math.round((1 - Number(a.gaze.off_screen_fraction ?? 0)) * 100)}%` : "eye contact not measured"} />
            {!shared && a.lipsync && <Chip icon={<Mic2 className="size-3" />} on={a.lipsync.measured} text={a.lipsync.measured ? `lip-sync: ${String(a.lipsync.decision ?? "")}` : "lip-sync not measured"} />}
            {!shared && a.voice && <Chip icon={<ShieldCheck className="size-3" />} on={a.voice.measured} text={a.voice.measured ? `voice match: ${String(a.voice.status ?? "")}` : "voice match not measured"} />}
          </div>
        </div>
      )}
    </Card>
  );
}

// ----------------------------------------------------------------------------- small pieces

function SectionTitle({ icon, title, sub }: { icon: React.ReactNode; title: string; sub?: string }) {
  return (
    <div className="mb-3">
      <h2 className="flex items-center gap-2 text-[18px] font-semibold"><span className="text-accent">{icon}</span>{title}</h2>
      {sub && <p className="mt-1 text-[13px] text-fg-muted">{sub}</p>}
    </div>
  );
}

function ScoreRing({ value }: { value: number | null }) {
  const v = value ?? 0;
  const c = 2 * Math.PI * 52;
  return (
    <div className="relative size-36">
      <svg viewBox="0 0 120 120" className="size-36 -rotate-90" aria-hidden>
        <circle cx="60" cy="60" r="52" fill="none" stroke="var(--surface-3)" strokeWidth="10" />
        <circle cx="60" cy="60" r="52" fill="none" stroke="var(--accent)" strokeWidth="10" strokeLinecap="round" strokeDasharray={`${v * c} ${c}`} />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <div className="text-center">
          <p className="font-mono text-[36px] font-semibold leading-none tabular" aria-label={`Overall ${pct(value)} out of 100`}>{pct(value)}</p>
          <p className="mt-1 text-[11px] text-fg-subtle">out of 100</p>
        </div>
      </div>
    </div>
  );
}

function DimBar({ label, value, compact }: { label: string; value: number | null; compact?: boolean }) {
  const w = value === null ? 0 : ((value - 1) / 4) * 100;
  return (
    <div className="flex items-center gap-3">
      <span className={cn("shrink-0 text-fg-muted", compact ? "w-32 text-[12px]" : "w-36 text-[13px]")}>{label}</span>
      <div
        className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3"
        {...(value === null
          ? { role: "img", "aria-label": `${label}: not measured` }
          : { role: "meter", "aria-label": label, "aria-valuemin": 1, "aria-valuemax": 5, "aria-valuenow": value })}
      >
        <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(value === null ? 0 : 4, w)}%` }} />
      </div>
      <span className="w-10 text-right font-mono text-[12px] tabular">{value === null ? "–" : value.toFixed(1)}</span>
    </div>
  );
}

function Trajectory({ values }: { values: number[] }) {
  if (!values.length) return <p className="text-[12px] text-fg-subtle">–</p>;
  const W = 220, H = 80, n = values.length;
  const x = (i: number) => (n === 1 ? W / 2 : 8 + (i * (W - 16)) / (n - 1));
  const y = (v: number) => H - 8 - ((v - 1) / 4) * (H - 16);
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-20 w-full" role="img" aria-label={`Difficulty per question: ${values.join(", ")}`}>
      {[1, 3, 5].map((g) => <line key={g} x1="0" x2={W} y1={y(g)} y2={y(g)} stroke="var(--line)" strokeDasharray="2 4" />)}
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="2" />
      {values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="3" fill="var(--accent)" />)}
    </svg>
  );
}

function List({ title, items, tone }: { title: string; items: string[]; tone: string }) {
  return (
    <div>
      <p className={cn("mb-1 text-[12px] font-medium", tone)}>{title}</p>
      <ul className="list-disc pl-5 text-[13px] leading-5 text-fg-muted">{items.map((t, i) => <li key={i}>{t}</li>)}</ul>
    </div>
  );
}

function Chip({ icon, on, text }: { icon: React.ReactNode; on: boolean; text: string }) {
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1", on ? "border-line text-fg" : "border-dashed border-line text-fg-subtle")}>{icon}{text}</span>;
}

function Collapsible({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group rounded-[16px] border border-line bg-surface">
      <summary className="flex cursor-pointer list-none items-center justify-between p-4 text-[15px] font-semibold">
        {title}<ChevronDown className="size-4 text-fg-subtle transition-transform group-open:rotate-180" />
      </summary>
      <div className="border-t border-line p-4">{children}</div>
    </details>
  );
}

/** The answer text with evidence quotes highlighted (case/punctuation-insensitive match). */
function Highlight({ text, quotes }: { text: string; quotes: string[] }) {
  const ranges: [number, number][] = [];
  const lower = text.toLowerCase();
  for (const q of quotes) {
    const i = lower.indexOf(q.toLowerCase().trim());
    if (q.trim() && i >= 0) ranges.push([i, i + q.trim().length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const out: React.ReactNode[] = [];
  let at = 0;
  ranges.forEach(([s, e], k) => {
    if (s < at) return;
    out.push(text.slice(at, s), <mark key={k} className="rounded bg-accent/20 px-0.5 text-fg">{text.slice(s, e)}</mark>);
    at = e;
  });
  out.push(text.slice(at));
  return <>{out}</>;
}
