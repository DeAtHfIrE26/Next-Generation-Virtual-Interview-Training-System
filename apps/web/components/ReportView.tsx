"use client";

import { DimensionBars } from "./Charts";

type Evidence = { dimension: string; quote: string; comment: string };
export type Report = {
  role: string; seniority: string; mode: string; generated_at: string;
  summary: { answers: number; overall: number | null; label: string; dimensions: Record<string, number | null>;
    words_per_minute: number | null; filler_per_100_words: number | null; off_screen_fraction: number | null };
  observations: string[];
  answers: { index: number; question: string; category: string; difficulty: number; follow_up: boolean; answer: string;
    scores: Record<string, number | null> | null; overall: number | null; method: string; label: string;
    evidence: Evidence[]; strengths: string[]; improvements: string[];
    delivery: { measured?: boolean; words_per_minute?: number | null; filler_per_100_words?: number; longest_pause_s?: number };
    gaze: { measured?: boolean; off_screen_fraction?: number; longest_off_screen_s?: number } }[];
  integrity?: { events: Record<string, number>; note: string };
  code?: { challenge_id: string; passed: number; total: number; summary: string }[];
};

const DIM_LABEL: Record<string, string> = { relevance: "Relevance", structure: "Structure", depth: "Depth",
  communication: "Communication", technical_accuracy: "Technical accuracy" };
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "not measured" : `${Math.round(v * 100)}%`);

export function ReportView({ r }: { r: Report }) {
  return (
    <div className="stack">
      <h1>Interview report</h1>
      <p className="muted">{r.role} · {r.seniority} · {new Date(r.generated_at).toLocaleString()}</p>
      <div className="grid three">
        <div className="card"><h3>Overall</h3><p style={{ fontSize: 32, margin: 0 }}>{pct(r.summary.overall)}</p>
          <span className="badge">{r.summary.label}</span></div>
        <div className="card"><h3>Speaking pace</h3><p style={{ fontSize: 24, margin: 0 }}>
          {r.summary.words_per_minute ? `${Math.round(r.summary.words_per_minute)} wpm` : "not measured"}</p>
          <span className="muted small">Fillers: {r.summary.filler_per_100_words ?? "not measured"} per 100 words</span></div>
        <div className="card"><h3>Looking at the screen</h3><p style={{ fontSize: 24, margin: 0 }}>
          {r.summary.off_screen_fraction === null ? "not measured" : pct(1 - r.summary.off_screen_fraction)}</p>
          <span className="muted small">of the time your face was visible</span></div>
      </div>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Rubric averages (1-5)</h2><span className="badge">{r.summary.label}</span>
        </div>
        <DimensionBars title="Rubric averages, 1 to 5" items={Object.entries(r.summary.dimensions).map(([k, v]) => ({ label: DIM_LABEL[k] ?? k, value: v }))} />
        <p className="small muted">Scores are labelled experimental until they have been validated against trained human raters (see the rubric). They describe the answers in this session, not you as a person.</p>
      </section>
      {r.observations.length > 0 && (
        <section className="card"><h2 style={{ marginTop: 0 }}>Observations</h2><ul>{r.observations.map((o) => <li key={o}>{o}</li>)}</ul></section>
      )}
      <h2>Answer by answer</h2>
      {r.answers.map((a) => (
        <article key={a.index} className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3 style={{ margin: 0 }}>Q{a.index + 1}{a.follow_up ? " (follow-up)" : ""}: {a.question}</h3>
            <span className="badge">{pct(a.overall)} · {a.method === "llm" ? "AI rubric" : "automatic"}</span>
          </div>
          <p className="muted small">{a.category.replace("_", " ")} · difficulty {a.difficulty}/5</p>
          <blockquote style={{ margin: 0 }}>{a.answer || <em className="muted">No answer recorded.</em>}</blockquote>
          {a.evidence.length > 0 && <ul className="small">{a.evidence.map((e, i) => (
            <li key={i}><mark className="quote">“{e.quote}”</mark> - {e.comment} <span className="muted">({e.dimension})</span></li>))}</ul>}
          <div className="grid two">
            <div><strong>Strengths</strong><ul>{a.strengths.map((s) => <li key={s}>{s}</li>)}</ul></div>
            <div><strong>To improve</strong><ul>{a.improvements.map((s) => <li key={s}>{s}</li>)}</ul></div>
          </div>
          <p className="small muted">
            Pace: {a.delivery.measured === false ? "not measured" : `${a.delivery.words_per_minute ?? "–"} wpm, longest pause ${a.delivery.longest_pause_s ?? 0}s`}
            {" · "}Looking away: {a.gaze.measured ? `${Math.round((a.gaze.off_screen_fraction ?? 0) * 100)}% of this answer` : "not measured"}
          </p>
        </article>
      ))}
      {r.code && r.code.length > 0 && (
        <section className="card"><h2 style={{ marginTop: 0 }}>Coding challenge</h2>
          {r.code.map((c, i) => <p key={i}>{c.challenge_id}: {c.summary}</p>)}</section>
      )}
      {r.integrity && Object.keys(r.integrity.events).length > 0 && (
        <section className="card"><h2 style={{ marginTop: 0 }}>Session notices</h2>
          <ul>{Object.entries(r.integrity.events).map(([k, n]) => <li key={k}>{k.replace(/_/g, " ")}: {n}</li>)}</ul>
          <p className="small muted">{r.integrity.note}</p></section>
      )}
    </div>
  );
}
