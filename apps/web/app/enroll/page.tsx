"use client";

// Optional E2/E3 enrolment: face (with a randomised liveness challenge) and voice (prompted
// phrases). Only encrypted templates are stored; the server refuses (503) when no licensed
// model is configured rather than pretending to verify.
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { api, ApiError, type Capabilities } from "@/lib/api";
import { MicCapture } from "@/lib/audio";
import { useUser } from "@/lib/hooks/useUser";
import { Captions } from "@/lib/speech";
import { faceCrop, roundLandmarks, VisionMonitor, type VisionSample } from "@/lib/vision";
import { encodeWav, toBase64 } from "@/lib/wav";

const STEP_TEXT: Record<string, string> = {
  blink: "Blink", turn_left: "Turn your head to your left", turn_right: "Turn your head to your right",
  open_mouth: "Open your mouth",
};

type Status = { face: boolean; voice: boolean; capabilities: Capabilities };

export default function Enroll() {
  const { user } = useUser();
  const [st, setSt] = useState<Status | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [prompt, setPrompt] = useState("");
  const video = useRef<HTMLVideoElement>(null);

  const refresh = () => api<Status>("/enrollment/status").then(setSt);
  useEffect(() => { if (user) void refresh(); }, [user]);
  if (!user) return null;

  async function enrolFace() {
    setBusy(true);
    setMsg("");
    let monitor: VisionMonitor | null = null;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.current!.srcObject = stream;
      await video.current!.play();
      monitor = await VisionMonitor.create();
      const ch = await api<{ nonce: string; steps: string[] }>("/enrollment/liveness/challenge", { method: "POST" });
      const frames: { t: number; landmarks: number[][] | null; faces: number }[] = [];
      const crops: string[] = [];
      let t0 = 0;
      monitor.start(video.current!, (s: VisionSample) => {
        t0 ||= s.t;
        frames.push({ t: (s.t - t0) / 1000, landmarks: s.landmarks ? roundLandmarks(s.landmarks) : null, faces: s.faces });
        if (s.landmarks && s.faces === 1 && crops.length < 6 && frames.length % 4 === 0) {
          const c = faceCrop(video.current!, s.landmarks);
          if (c) crops.push(c);
        }
      }, 15, 0.0001);
      setPrompt("Look at the camera and keep still…");
      await new Promise((r) => setTimeout(r, 1200));
      for (const step of ch.steps) {
        setPrompt(STEP_TEXT[step] ?? step);
        await new Promise((r) => setTimeout(r, 1800));
        setPrompt("Back to centre");
        await new Promise((r) => setTimeout(r, 900));
      }
      monitor.stop();
      setPrompt("");
      stream.getTracks().forEach((t) => t.stop());
      await api("/enrollment/face", { method: "POST", json: { nonce: ch.nonce, series: { nonce: ch.nonce, frames }, faces: crops } });
      setMsg("Face check set up.");
      await refresh();
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Face setup failed. Check camera permissions and lighting.");
    } finally {
      monitor?.close();
      setBusy(false);
      setPrompt("");
    }
  }

  async function enrolVoice() {
    setBusy(true);
    setMsg("");
    const recordings: { nonce: string; audio_wav: string; client_transcript: string }[] = [];
    let mic: MicCapture | null = null;
    try {
      mic = await MicCapture.open();
      for (let i = 0; i < 3; i++) {
        const p = await api<{ nonce: string; phrase: string }>("/enrollment/voice/phrase", { method: "POST" });
        setPrompt(`Read aloud: “${p.phrase}”`);
        const cap = Captions.supported() ? new Captions(() => undefined) : null;
        cap?.start();
        mic.startRecording();
        await new Promise((r) => setTimeout(r, 4500));
        const rec = mic.stopRecording();
        const heard = cap?.stop() ?? "";
        recordings.push({ nonce: p.nonce, audio_wav: toBase64(encodeWav(rec.samples, rec.sampleRate)), client_transcript: heard });
      }
      setPrompt("");
      await api("/enrollment/voice", { method: "POST", json: { recordings } });
      setMsg("Voice check set up.");
      await refresh();
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Voice setup failed. Check microphone permissions.");
    } finally {
      await mic?.close();
      setBusy(false);
      setPrompt("");
    }
  }

  const caps = st?.capabilities;
  return (
    <div className="stack" style={{ maxWidth: 760 }}>
      <h1>Optional identity checks</h1>
      <p className="lead">These confirm the same person stays in the session. They are optional for practice. We store only an encrypted numeric template, never your photos or recordings, and you can delete it any time.</p>
      <p className="small muted">Requires the matching permission in <Link href="/settings/privacy">Privacy settings</Link>.</p>
      <div className="grid two">
        <div className="card stack">
          <h3>Face {st?.face && <span className="badge">set up</span>}</h3>
          {caps && !caps.face_verification && <p className="notice small">Face verification is not enabled on this server yet.</p>}
          <video ref={video} className="self" muted playsInline aria-label="Camera preview" />
          <button className="btn primary" disabled={busy || !caps?.face_verification} onClick={() => void enrolFace()}>Set up face check</button>
        </div>
        <div className="card stack">
          <h3>Voice {st?.voice && <span className="badge">set up</span>}</h3>
          {caps && !caps.voice_verification && <p className="notice small">Voice verification is not enabled on this server yet.</p>}
          <p className="muted small">You will read three short random phrases.</p>
          <button className="btn primary" disabled={busy || !caps?.voice_verification} onClick={() => void enrolVoice()}>Set up voice check</button>
        </div>
      </div>
      {prompt && <p className="notice" style={{ fontSize: "1.3rem" }} aria-live="assertive">{prompt}</p>}
      {msg && <p className="notice" role="status">{msg}</p>}
    </div>
  );
}
