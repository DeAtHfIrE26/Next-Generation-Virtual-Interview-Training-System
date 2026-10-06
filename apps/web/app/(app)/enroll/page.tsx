"use client";

// Optional E2/E3 enrolment: face (with a randomised liveness challenge) and voice (prompted
// phrases). Only encrypted templates are stored; the server refuses (503) when no licensed
// model is configured rather than pretending to verify.
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, Camera, Check, Lock, Mic, ShieldCheck } from "lucide-react";
import { Alert, Badge, Button, Card, cn, Skeleton } from "@/components/ui";
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
  const [msgOk, setMsgOk] = useState(false);
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState<"face" | "voice" | null>(null);
  const [prompt, setPrompt] = useState("");
  const video = useRef<HTMLVideoElement>(null);

  const refresh = () => api<Status>("/enrollment/status").then(setSt);
  useEffect(() => { if (user) void api<Status>("/enrollment/status").then(setSt); }, [user]);

  async function enrolFace() {
    setBusy(true);
    setActive("face");
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
      setMsgOk(true);
      await refresh();
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Face setup failed. Check camera permissions and lighting.");
      setMsgOk(false);
    } finally {
      monitor?.close();
      setBusy(false);
      setActive(null);
      setPrompt("");
    }
  }

  async function enrolVoice() {
    setBusy(true);
    setActive("voice");
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
      setMsgOk(true);
      await refresh();
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Voice setup failed. Check microphone permissions.");
      setMsgOk(false);
    } finally {
      await mic?.close();
      setBusy(false);
      setActive(null);
      setPrompt("");
    }
  }

  const caps = st?.capabilities;
  const loading = !user || !st;

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-8">
      <header className="max-w-[760px]">
        <p className="text-[13px] font-medium text-accent">Optional</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9">Optional identity checks</h1>
        <p className="mt-3 text-[15px] leading-6 text-fg-muted">These confirm the same person stays in the session. They are optional for practice. We store only an encrypted numeric template, never your photos or recordings, and you can delete it any time.</p>
      </header>

      <Card as="section" aria-labelledby="step-consent" className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center">
        <StepNumber n={1} />
        <div className="flex-1">
          <h2 id="step-consent" className="text-[15px] font-semibold">Grant the matching permission</h2>
          <p className="mt-0.5 text-[13px] text-fg-muted">
            Requires the matching permission in <Link href="/settings/privacy" className="text-fg underline decoration-line-strong underline-offset-2 hover:decoration-accent">Privacy settings</Link>.
          </p>
        </div>
        <Link href="/settings/privacy" className="inline-flex items-center gap-1 text-[13px] font-medium text-accent hover:text-accent-strong">
          Open Privacy settings <ArrowRight className="size-3.5" aria-hidden />
        </Link>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Face */}
        <Card as="section" aria-labelledby="step-face" className="flex flex-col gap-4 p-5">
          <div className="flex items-start gap-3">
            <StepNumber n={2} />
            <div className="flex-1">
              <h2 id="step-face" className="flex flex-wrap items-center gap-2 text-[15px] font-semibold">
                <Camera className="size-4 text-accent" aria-hidden /> Face
                {loading ? null : <StateBadge done={!!st?.face} available={!!caps?.face_verification} running={active === "face"} />}
              </h2>
              <p className="mt-0.5 text-[13px] text-fg-muted">A short liveness challenge: follow the prompts on screen.</p>
            </div>
          </div>
          {caps && !caps.face_verification && <Alert tone="warn">Face verification is not enabled on this server yet.</Alert>}
          <div className="relative aspect-[4/3] w-full overflow-hidden rounded-[12px] border border-line bg-surface-2">
            <video ref={video} className={cn("size-full -scale-x-100 object-cover", active !== "face" && "invisible")} muted playsInline aria-label="Camera preview" />
            {active !== "face" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-fg-subtle">
                <Camera className="size-6" aria-hidden />
                <span className="text-[13px]">Camera preview appears here</span>
              </div>
            )}
            {active === "face" && prompt && (
              <div className="absolute inset-x-3 bottom-3 rounded-[12px] border border-line bg-surface/90 px-4 py-3 text-center text-[18px] font-semibold leading-7 text-fg shadow-[var(--shadow-float)] backdrop-blur" aria-hidden>
                {prompt}
              </div>
            )}
          </div>
          {loading ? <Skeleton className="h-10" /> : (
            <Button disabled={busy || !caps?.face_verification} loading={active === "face"} onClick={() => void enrolFace()}>
              {active === "face" ? "Setting up face check…" : st?.face ? "Set up face check again" : "Set up face check"}
            </Button>
          )}
        </Card>

        {/* Voice */}
        <Card as="section" aria-labelledby="step-voice" className="flex flex-col gap-4 p-5">
          <div className="flex items-start gap-3">
            <StepNumber n={3} />
            <div className="flex-1">
              <h2 id="step-voice" className="flex flex-wrap items-center gap-2 text-[15px] font-semibold">
                <Mic className="size-4 text-accent" aria-hidden /> Voice
                {loading ? null : <StateBadge done={!!st?.voice} available={!!caps?.voice_verification} running={active === "voice"} />}
              </h2>
              <p className="mt-0.5 text-[13px] text-fg-muted">You will read three short random phrases.</p>
            </div>
          </div>
          {caps && !caps.voice_verification && <Alert tone="warn">Voice verification is not enabled on this server yet.</Alert>}
          <div className={cn("flex min-h-[160px] flex-1 flex-col items-center justify-center gap-3 rounded-[12px] border px-5 py-6 text-center", active === "voice" ? "border-live/40 bg-live/[0.06]" : "border-line bg-surface-2")}>
            {active === "voice" ? (
              <>
                <span className="flex items-center gap-2 text-[12px] font-medium uppercase tracking-[0.06em] text-live">
                  <span className="size-2 rounded-full bg-live animate-pulse-ring" aria-hidden /> Recording
                </span>
                <p className="text-[18px] font-semibold leading-7 text-fg" aria-hidden>{prompt || "Preparing…"}</p>
              </>
            ) : (
              <>
                <Mic className="size-6 text-fg-subtle" aria-hidden />
                <span className="text-[13px] text-fg-subtle">Phrases appear here, one at a time</span>
              </>
            )}
          </div>
          {loading ? <Skeleton className="h-10" /> : (
            <Button disabled={busy || !caps?.voice_verification} loading={active === "voice"} onClick={() => void enrolVoice()}>
              {active === "voice" ? "Setting up voice check…" : st?.voice ? "Set up voice check again" : "Set up voice check"}
            </Button>
          )}
        </Card>
      </div>

      {/* Screen-reader announcement of each prompt (the visual copies above are aria-hidden). */}
      <p className="sr-only" aria-live="assertive">{prompt}</p>
      {msg && <Alert tone={msgOk ? "success" : "danger"}>{msg}</Alert>}

      <p className="flex items-start gap-2 text-[13px] leading-5 text-fg-muted">
        <Lock className="mt-0.5 size-3.5 shrink-0" aria-hidden />
        <span>Delete your templates any time from <Link href="/settings/privacy" className="text-fg underline decoration-line-strong underline-offset-2 hover:decoration-accent">Privacy settings</Link>.</span>
      </p>
    </div>
  );
}

function StepNumber({ n }: { n: number }) {
  return (
    <span className="flex size-7 shrink-0 items-center justify-center rounded-full border border-line bg-surface-2 font-mono text-[12px] font-medium text-fg-muted" aria-hidden>
      {n}
    </span>
  );
}

function StateBadge({ done, available, running }: { done: boolean; available: boolean; running: boolean }) {
  if (running) return <Badge tone="live">in progress</Badge>;
  if (done) return <Badge tone="success"><Check className="size-3" aria-hidden /> set up</Badge>;
  if (!available) return <Badge tone="warn">unavailable</Badge>;
  return <Badge><ShieldCheck className="size-3" aria-hidden /> not set up</Badge>;
}
