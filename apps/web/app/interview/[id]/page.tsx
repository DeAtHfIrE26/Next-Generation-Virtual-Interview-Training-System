"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Avatar, NeuralAvatar } from "@/components/Avatar";
import { api, ApiError, type Capabilities } from "@/lib/api";
import { MicCapture } from "@/lib/audio";
import { useUser } from "@/lib/hooks/useUser";
import { Captions, playServerAudio, speakBrowser, type Speaking } from "@/lib/speech";
import { Vad } from "@/lib/vad";
import { faceCrop, VisionMonitor, type VisionSample } from "@/lib/vision";
import type { Timeline } from "@/lib/visemes";
import { encodeWav, toBase64 } from "@/lib/wav";

type Challenge = { id: string; title: string; prompt: string; languages: string[]; starter: Record<string, string>;
  examples: { stdin: string; expected: string }[]; hidden_tests: number };
type Question = { index: number; question: string; category: string; difficulty: number; follow_up: boolean;
  source: string; challenge?: Challenge; speech?: { audio_b64: string; mime: string; visemes: [number, string][] }; done?: boolean };
type Notice = { event: string; message: string; end_session: boolean };
type Phase = "setup" | "asking" | "listening" | "processing" | "feedback" | "finishing" | "ended";

const VISION_ENABLED = process.env.NEXT_PUBLIC_VISION !== "off";

export default function Room() {
  const { id } = useParams<{ id: string }>();
  const { user } = useUser();
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>("setup");
  const [q, setQ] = useState<Question | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [timeline, setTimeline] = useState<Timeline>([]);
  const [speakStart, setSpeakStart] = useState<number | null>(null);
  const [caption, setCaption] = useState({ interim: "", final: "" });
  const [typed, setTyped] = useState("");
  const [feedback, setFeedback] = useState<{ summary: string; overall: number; label: string; improvements: string[] } | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);
  const [status, setStatus] = useState({ camera: "off", mic: "off", vision: "off" });
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [error, setError] = useState("");
  const [neural, setNeural] = useState<string | null>(null); // stream URL when the neural avatar is used

  const videoRef = useRef<HTMLVideoElement>(null);
  const mic = useRef<MicCapture | null>(null);
  const vision = useRef<VisionMonitor | null>(null);
  const captions = useRef<Captions | null>(null);
  const speaking = useRef<Speaking | null>(null);
  const vad = useRef(new Vad());
  const phaseRef = useRef<Phase>("setup");
  const recStart = useRef(0);
  const mouth = useRef<{ times: number[]; values: (number | null)[] }>({ times: [], values: [] });
  const gaze = useRef<[number, boolean | null][]>([]);
  const obs = useRef<{ t: number; type: string; present: boolean }[]>([]);
  const lastLandmarks = useRef<VisionSample["landmarks"]>(null);
  const questionAt = useRef(0);
  const heardSpeech = useRef(false);

  const setPhaseBoth = (p: Phase) => { phaseRef.current = p; setPhase(p); };

  // ------------------------------------------------------------------ media setup
  const startMedia = useCallback(async () => {
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true });
      setStatus((s) => ({ ...s, camera: "on", mic: "on" }));
    } catch {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setStatus((s) => ({ ...s, mic: "on", camera: "unavailable" }));
      } catch {
        setStatus({ camera: "unavailable", mic: "unavailable", vision: "off" });
      }
    }
    if (stream && videoRef.current && stream.getVideoTracks().length) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => undefined);
    }
    if (stream && stream.getAudioTracks().length) {
      mic.current = await MicCapture.open(new MediaStream(stream.getAudioTracks()));
      mic.current.onFrame = (db, t) => {
        const ev = vad.current.push(db, t, phaseRef.current === "asking");
        if (!ev) return;
        if (ev.type === "speech_start" && phaseRef.current === "asking") { speaking.current?.cancel(); beginListening(); }
        if (ev.type === "speech_start" && phaseRef.current === "listening") heardSpeech.current = true;
        if (ev.type === "speech_end" && phaseRef.current === "listening" && heardSpeech.current && ev.durationMs > 800) void submitAnswer();
      };
    }
    if (VISION_ENABLED && stream?.getVideoTracks().length) {
      try {
        vision.current = await VisionMonitor.create();
        setStatus((s) => ({ ...s, vision: vision.current!.canDetectPhones ? "on" : "faces only" }));
        vision.current.start(videoRef.current!, onVision);
      } catch {
        setStatus((s) => ({ ...s, vision: "unavailable" }));
      }
    }
    if (Captions.supported()) captions.current = new Captions((interim, final) => setCaption({ interim, final }));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function onVision(s: VisionSample) {
    lastLandmarks.current = s.landmarks;
    const t = s.t / 1000;
    obs.current.push({ t, type: "no_face", present: s.faces === 0 }, { t, type: "second_person", present: s.faces > 1 });
    if (s.phone !== null) obs.current.push({ t, type: "phone", present: s.phone });
    if (phaseRef.current === "listening") {
      mouth.current.times.push((s.t - recStart.current) / 1000);
      mouth.current.values.push(s.aperture);
      gaze.current.push([(s.t - recStart.current) / 1000, s.onScreen]);
    }
  }

  // Flush integrity observations every 2 s (server debounces and applies the session policy).
  useEffect(() => {
    const iv = setInterval(async () => {
      if (!obs.current.length || phaseRef.current === "ended" || phaseRef.current === "setup") return;
      const batch = obs.current.splice(0, obs.current.length).slice(-500);
      try {
        const r = await api<{ notices: Notice[]; status: string }>(`/sessions/${id}/events`, { method: "POST", json: { observations: batch } });
        if (r.notices.length) setNotices((n) => [...r.notices, ...n].slice(0, 5));
        if (r.status === "ended_by_policy") end();
      } catch { /* transient: next batch retries */ }
    }, 2000);
    return () => clearInterval(iv);
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Periodic face verification (only when the server can verify and the user enrolled).
  useEffect(() => {
    if (!caps?.face_verification) return;
    const iv = setInterval(async () => {
      const lm = lastLandmarks.current;
      if (!lm || !videoRef.current || phaseRef.current === "ended") return;
      const img = faceCrop(videoRef.current, lm);
      if (!img) return;
      const r = await api<{ notice: Notice | null }>(`/sessions/${id}/face-check`, { method: "POST", json: { image: img, t: performance.now() / 1000 } }).catch(() => null);
      if (r?.notice) setNotices((n) => [r.notice!, ...n].slice(0, 5));
    }, 45000);
    return () => clearInterval(iv);
  }, [caps, id]);

  useEffect(() => () => { vision.current?.close(); void mic.current?.close(); speaking.current?.cancel(); }, []);

  // ------------------------------------------------------------------ flow
  async function start() {
    setError("");
    await startMedia();
    const s = await api<{ capabilities: Capabilities; status: string }>(`/sessions/${id}`);
    setCaps(s.capabilities);
    if (s.status !== "active") { router.replace(`/reports/${id}`); return; }
    await nextQuestion();
  }

  async function nextQuestion() {
    setFeedback(null);
    setTyped("");
    setCaption({ interim: "", final: "" });
    setPhaseBoth("processing");
    try {
      const nq = await api<Question>(`/sessions/${id}/next`, { method: "POST" });
      if (nq.done) { await finish(); return; }
      setQ(nq);
      if (nq.challenge) setChallenge(nq.challenge);
      questionAt.current = performance.now();
      void ask(nq);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) end();
      else setError(e instanceof ApiError ? e.message : "Network problem; try again.");
    }
  }

  async function tryNeural(nq: Question): Promise<string | null> {
    // Degradation chain: neural talking head (if enabled and fast) -> viseme rig -> captions.
    if (!caps?.neural_avatar) return null;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 1200);
    try {
      const r = await api<{ stream_url: string }>("/avatar/render", { method: "POST", json: { text: nq.question, session_id: id }, signal: ctl.signal });
      return r.stream_url;
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  async function ask(nq: Question) {
    setPhaseBoth("asking");
    const stream = await tryNeural(nq);
    setNeural(stream);
    if (stream) return; // the neural stream carries its own audio; onEnded moves to listening
    const onStart = () => setSpeakStart(performance.now());
    speaking.current = nq.speech
      ? playServerAudio(nq.speech.audio_b64, nq.speech.mime, nq.speech.visemes, setTimeline, onStart)
      : speakBrowser(nq.question, setTimeline, onStart);
    void speaking.current.done.then(() => {
      setSpeakStart(null);
      if (phaseRef.current === "asking") beginListening();
    });
  }

  function beginListening() {
    setSpeakStart(null);
    setPhaseBoth("listening");
    heardSpeech.current = false;
    mouth.current = { times: [], values: [] };
    gaze.current = [];
    recStart.current = mic.current?.startRecording() ?? performance.now();
    captions.current?.start();
  }

  async function submitAnswer() {
    if (phaseRef.current !== "listening") return;
    setPhaseBoth("processing");
    const spoken = captions.current?.stop() ?? "";
    const rec = mic.current?.stopRecording();
    const body: Record<string, unknown> = { transcript: (typed || spoken || caption.final).trim() };
    if (rec && rec.samples.length > 1600) body.audio_wav = toBase64(encodeWav(rec.samples, rec.sampleRate));
    if (mouth.current.times.length > 10) body.mouth = mouth.current;
    if (gaze.current.length > 10) body.gaze_samples = gaze.current;
    try {
      const r = await api<{ evaluation: { summary: string; overall: number; label: string; improvements: string[] };
        notices: Notice[]; status: string }>(`/sessions/${id}/answer`, { method: "POST", json: body });
      setFeedback(r.evaluation);
      if (r.notices.length) setNotices((n) => [...r.notices, ...n].slice(0, 5));
      if (r.status === "ended_by_policy") { end(); return; }
      setPhaseBoth("feedback");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not submit the answer.");
      setPhaseBoth("listening");
    }
  }

  async function finish() {
    setPhaseBoth("finishing");
    await api(`/sessions/${id}/finish`, { method: "POST" });
    router.push(`/reports/${id}`);
  }

  function end() {
    setPhaseBoth("ended");
    speaking.current?.cancel();
  }

  const reportLatency = useCallback((t: number) => {
    void api("/metrics/latency", { method: "POST", json: { metric: "question_to_first_avatar_frame", ms: Math.round(t - questionAt.current) } }).catch(() => undefined);
  }, []);

  if (!user) return null;
  return (
    <div className="stack">
      {phase === "setup" && (
        <div className="card stack" style={{ maxWidth: 720 }}>
          <h1>Ready?</h1>
          <p className="muted">We&apos;ll ask for your camera and microphone. Camera analysis (face, gaze, phone) runs in this browser; only numbers are sent. You can also type your answers.</p>
          <button className="btn primary" onClick={() => void start()}>Start the interview</button>
        </div>
      )}
      {error && <p className="notice error" role="alert">{error}</p>}
      {phase === "ended" && (
        <div className="notice error" role="alert">
          The session ended because of repeated integrity notices (strict practice mode). <a href={`/reports/${id}`} onClick={(e) => { e.preventDefault(); void finish(); }}>See your report</a>
        </div>
      )}
      <div className="room" hidden={phase === "setup"}>
        <section className="card stack" aria-live="polite">
          {neural && phase === "asking" ? (
            <NeuralAvatar src={neural} onError={() => { setNeural(null); if (q) speaking.current = speakBrowser(q.question, setTimeline, () => setSpeakStart(performance.now())); }}
              onEnded={() => { setNeural(null); beginListening(); }} onPlaying={() => reportLatency(performance.now())} />
          ) : (
            <Avatar timeline={timeline} startedAt={speakStart} onFirstFrame={reportLatency} />
          )}
          {q && (
            <div>
              <p className="small muted">Question {q.index + 1}{q.follow_up ? " · follow-up" : ""} · {q.category.replace("_", " ")} · difficulty {q.difficulty}/5</p>
              <p style={{ fontSize: "1.15rem" }}>{q.question}</p>
            </div>
          )}
          {phase === "asking" && <button className="btn" onClick={() => { speaking.current?.cancel(); beginListening(); }}>Skip to answering</button>}
          {phase === "listening" && (
            <div className="stack">
              <p className="caption" aria-live="polite">{caption.final} <span className="muted">{caption.interim}</span></p>
              <label className="field"><span>{Captions.supported() ? "Or type your answer" : "Type your answer (speech captions are not supported in this browser)"}</span>
                <textarea rows={4} value={typed} onChange={(e) => setTyped(e.target.value)} /></label>
              <button className="btn primary" onClick={() => void submitAnswer()}>Done answering</button>
            </div>
          )}
          {phase === "processing" && <p className="muted">Thinking…</p>}
          {phase === "feedback" && feedback && (
            <div className="stack">
              <p><strong>{Math.round(feedback.overall * 100)}%</strong> <span className="badge">{feedback.label}</span></p>
              <p>{feedback.summary}</p>
              {feedback.improvements[0] && <p className="muted">Try next time: {feedback.improvements[0]}</p>}
              <button className="btn primary" onClick={() => void nextQuestion()}>Next question</button>
            </div>
          )}
          {phase === "finishing" && <p className="muted">Building your report…</p>}
        </section>
        <aside className="stack">
          <video ref={videoRef} className="self" muted playsInline aria-label="Your camera" />
          <p className="small muted">Camera: {status.camera} · Mic: {status.mic} · On-device vision: {status.vision}
            {caps && <> · Feedback: {caps.llm ? "AI rubric" : "automatic (offline)"}</>}</p>
          {notices.map((n, i) => <p key={i} className="notice small" role="status">{n.message}</p>)}
          {challenge && <CodePanel sessionId={id} challenge={challenge} />}
        </aside>
      </div>
    </div>
  );
}

function CodePanel({ sessionId, challenge }: { sessionId: string; challenge: Challenge }) {
  const [lang, setLang] = useState(challenge.languages[0] ?? "python");
  const [src, setSrc] = useState(challenge.starter[challenge.languages[0] ?? "python"] ?? "");
  const [out, setOut] = useState<string>("");
  async function run(final: boolean) {
    setOut("Running…");
    try {
      const r = await api<{ summary: string; error: string | null; outcomes: { passed: boolean; hidden: boolean; status: string; stdout: string | null; expected: string | null }[] }>(
        `/sessions/${sessionId}/code`, { method: "POST", json: { language: lang, source: src, final } });
      setOut([r.summary, ...r.outcomes.map((o, i) => `Test ${i + 1}${o.hidden ? " (hidden)" : ""}: ${o.passed ? "passed" : o.status}${!o.hidden && !o.passed ? `\n  expected: ${o.expected}\n  got: ${o.stdout}` : ""}`)].join("\n"));
    } catch (e) {
      setOut(e instanceof ApiError ? e.message : "Could not run code.");
    }
  }
  return (
    <section className="card stack">
      <h3>Coding challenge: {challenge.title}</h3>
      <p className="small">{challenge.prompt}</p>
      {challenge.examples.map((e, i) => <pre key={i} className="small">input:{"\n"}{e.stdin || "(none)"}{"\n"}expected:{"\n"}{e.expected}</pre>)}
      <p className="small muted">{challenge.hidden_tests} hidden test(s) are also run.</p>
      <select aria-label="Language" value={lang} onChange={(e) => { setLang(e.target.value); setSrc(challenge.starter[e.target.value] ?? ""); }}>
        {challenge.languages.map((l) => <option key={l}>{l}</option>)}
      </select>
      <textarea className="code" aria-label="Code editor" spellCheck={false} value={src} onChange={(e) => setSrc(e.target.value)} />
      <div className="row"><button className="btn" onClick={() => void run(false)}>Run tests</button><button className="btn primary" onClick={() => void run(true)}>Submit</button></div>
      {out && <pre className="small" aria-live="polite">{out}</pre>}
    </section>
  );
}
