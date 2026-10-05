import Link from "next/link";

export default function Home() {
  return (
    <div className="stack">
      <h1>Practise the interview before the interview</h1>
      <p className="lead">
        An AI interviewer asks questions tailored to your role, seniority and resume, adapts to your answers,
        and gives feedback you can check: every score points to the words you said.
      </p>
      <div className="row">
        <Link className="btn primary" href="/signup">Start practising free</Link>
        <Link className="btn" href="/login">Sign in</Link>
      </div>
      <div className="grid three" style={{ marginTop: 24 }}>
        <div className="card"><h3>Adaptive questions</h3><p className="muted">Questions follow your resume and job description, get harder when you do well, and follow up when an answer is thin.</p></div>
        <div className="card"><h3>Explainable feedback</h3><p className="muted">Relevance, structure, depth and clarity, each with quotes from your answer. Pace, pauses and filler words with timestamps.</p></div>
        <div className="card"><h3>Your data, your call</h3><p className="muted">Camera analysis runs in your browser. Identity checks are optional, stored only as encrypted templates, and deletable any time.</p></div>
      </div>
      <p className="small muted">Scores are labelled experimental until they have been validated against human interviewers. Accuracy figures, when published, come only from our evaluation report.</p>
    </div>
  );
}
