"use client";

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { AlertTriangle, Clock, Keyboard, Mic, MicOff, PhoneOff, Repeat2, SkipForward, Wifi, WifiOff } from "lucide-react";
import { AvatarStage } from "@/components/avatar/AvatarStage";
import { CodePanel, type Challenge } from "@/components/room/CodePanel";
import { DeviceCheck, type DeviceChoice } from "@/components/room/DeviceCheck";
import { DiagPanel } from "@/components/room/DiagPanel";
import { BrandMark } from "@/components/shell";
import { Alert, Badge, Button, Card, cn, Dialog, Spinner, Textarea } from "@/components/ui";
import { detectQuality, type Speaker } from "@/lib/avatar/speaker";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";
import { RoomController, type RoomState } from "@/lib/room/controller";
import { stopStream } from "@/lib/voice/mic";
import type { Persona } from "@/lib/voice/realtime";
import { VisionMonitor } from "@/lib/vision";

interface SessionInfo {
  id: string;
  status: string;
  params: { role: string; company?: string; interview_type: string; round: string; duration_minutes: number };
  persona: Persona;
  blueprint: { competencies: { id: string; name: string; minutes: number; questions: number }[]; emergency: boolean } | null;
  challenge: Challenge | null;
  capabilities: { face_verification: boolean };
}

const VISION_ENABLED = process.env.NEXT_PUBLIC_VISION !== "off";

export default function RoomPage() {
  return (
    <Suspense fallback={<FullScreen><Spinner /></FullScreen>}>
      <Room />
    </Suspense>
  );
}

function Room() {
  const { id } = useParams<{ id: string }>();
  const debug = useSearchParams().get("debug") === "1";
  const router = useRouter();
  useUser();
  const [info, setInfo] = useState<SessionInfo | null>(null);
  const [loadError, setLoadError] = useState("");
  const [speakerReady, setSpeakerReady] = useState(false);
  const [joining, setJoining] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [typing, setTyping] = useState(false);
  const [confirmEnd, setConfirmEnd] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const vision = useRef<VisionMonitor | null>(null);

  const finish = useCallback(async () => {
    setFinishing(true);
    try {
      await api(`/sessions/${id}/finish`, { method: "POST" });
    } catch {
      /* the report page retries finishing */
    }
    router.push(`/reports/${id}`);
  }, [id, router]);

  const ctl = useMemo(() => new RoomController(id, { faceVerification: false, onDone: () => void finish() }), [id, finish]);
  const state = useSyncExternalStore(ctl.subscribe, ctl.getSnapshot, ctl.getSnapshot);
  useEffect(() => { if (debug) (window as unknown as { __room?: RoomController }).__room = ctl; }, [debug, ctl]);

  useEffect(() => {
    api<SessionInfo>(`/sessions/${id}`)
      .then((s) => {
        if (s.status !== "active") router.replace(`/reports/${id}`);
        else setInfo(s);
      })
      .catch((e) => setLoadError(e instanceof ApiError ? e.message : "Could not load this interview."));
  }, [id, router]);

  useEffect(() => () => {
    void ctl.destroy();
    vision.current?.close();
    vision.current = null;
  }, [ctl]);
  useEffect(() => () => stopStream(stream), [stream]);

  const onSpeaker = useCallback((sp: Speaker) => { ctl.attachSpeaker(sp); setSpeakerReady(true); }, [ctl]);

  async function join(choice: DeviceChoice) {
    setJoining(true);
    setStream(choice.stream);
    ctl.setFaceVerification(!!info?.capabilities.face_verification);
    await ctl.join(choice.stream);
    setJoining(false);
    if (!choice.stream) setTyping(true);
  }

  // self view + on-device vision once joined with a camera
  useEffect(() => {
    const v = video.current;
    if (!state.joined || !v || !stream?.getVideoTracks().length) return;
    v.srcObject = stream;
    void v.play().catch(() => undefined);
    ctl.setVideo(v);
    if (!VISION_ENABLED) return;
    let cancelled = false;
    VisionMonitor.create()
      .then((m) => {
        if (cancelled) return m.close();
        vision.current = m;
        const soft = detectQuality().tier === "software";
        m.start(v, (s) => ctl.onVision(s), soft ? 4 : 12, soft ? 0.5 : 2);
      })
      .catch((e) => console.warn("on-device vision unavailable", e));
    return () => {
      cancelled = true;
      vision.current?.close();
      vision.current = null;
    };
  }, [state.joined, stream, ctl]);

  // keyboard: M mute, T type, Esc closes typing
  useEffect(() => {
    if (!state.joined) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
      if (e.key === "m" || e.key === "M") ctl.setMuted(!ctl.getSnapshot().muted);
      if (e.key === "t" || e.key === "T") { e.preventDefault(); setTyping(true); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.joined, ctl]);

  if (loadError) return <FullScreen><Alert tone="danger" title="Couldn't open the interview">{loadError}</Alert></FullScreen>;
  if (!info) return <FullScreen><Spinner /></FullScreen>;

  const persona = state.persona ?? info.persona;
  const avatarState = state.joined ? ctl.avatarState() : "idle";

  return (
    <div className="flex min-h-dvh flex-col bg-canvas">
      <TopBar info={info} state={state} persona={persona} />

      <main className={cn("mx-auto grid w-full max-w-[1400px] flex-1 gap-4 px-4 pb-28 pt-4", state.joined ? "lg:grid-cols-[1fr_380px]" : "lg:grid-cols-[1fr_1fr] lg:items-start lg:pt-6")}>
        {/* Stage (same element before and after joining, so the avatar is loaded once) */}
        <section className="relative flex min-h-[340px] flex-col" data-testid="stage" data-audio={state.audioPlaying ? "playing" : "idle"}>
          <AvatarStage
            look={persona.look}
            state={avatarState}
            onSpeaker={onSpeaker}
            name={`${persona.name} · ${persona.title}`}
            className={cn("w-full", state.joined ? "min-h-[360px] flex-1 lg:min-h-[560px]" : "aspect-[4/3] lg:aspect-auto lg:h-[min(760px,calc(100dvh-130px))]")}
          />
          {state.joined && (
            <>
              {state.current?.emergency && (
                <div className="absolute right-4 top-4" data-testid="emergency-badge">
                  <Badge tone="warn"><AlertTriangle className="size-3" /> Backup question: the AI interviewer is unavailable</Badge>
                </div>
              )}
              <Captions state={state} persona={persona.name} />
              {stream?.getVideoTracks().length ? (
                <video ref={video} muted playsInline aria-label="Your camera" className="absolute bottom-4 right-4 hidden aspect-video w-44 -scale-x-100 rounded-[12px] border border-line object-cover shadow-[var(--shadow-float)] sm:block" />
              ) : null}
            </>
          )}
        </section>

        {!state.joined ? (
          <Card className="flex flex-col gap-5 p-6">
            <div>
              <p className="text-[13px] font-medium text-accent">Before you join</p>
              <h1 className="mt-1 text-[24px] font-semibold leading-8">Check your camera and microphone</h1>
              <p className="mt-1 text-[14px] text-fg-muted">
                {info.params.role}{info.params.company ? ` at ${info.params.company}` : ""} · {label(info.params.interview_type)} · {info.params.duration_minutes} min.
                Speak naturally; {persona.name} waits for you to finish. You can interrupt at any time.
              </p>
            </div>
            <DeviceCheck onJoin={(c) => void join(c)} joining={joining} ready={speakerReady} />
          </Card>
        ) : (
          <SidePanel info={info} state={state} sessionId={id} />
        )}
      </main>

      {state.joined && (
        <Dock
          state={state}
          hasMic={!!stream?.getAudioTracks().length}
          onMute={() => ctl.setMuted(!state.muted)}
          onType={() => setTyping((t) => !t)}
          onRepeat={() => ctl.repeat()}
          onSkip={() => ctl.skip()}
          onEnd={() => setConfirmEnd(true)}
        />
      )}

      {typing && state.joined && <TypeBox onSend={(t) => { ctl.sendText(t); if (stream) setTyping(false); }} onClose={() => setTyping(false)} disabled={state.phase === "thinking" || state.phase === "done"} />}

      {state.error && state.joined && (
        <div className="fixed left-1/2 top-16 z-40 w-[min(92vw,520px)] -translate-x-1/2">
          <Alert tone="warn" title={errorTitle(state.error.code)} action={<Button size="sm" variant="ghost" onClick={() => ctl.dismissError()}>Dismiss</Button>}>
            {state.error.message}
          </Alert>
        </div>
      )}

      <Dialog
        open={confirmEnd}
        onClose={() => setConfirmEnd(false)}
        title="End the interview?"
        footer={<><Button variant="secondary" onClick={() => setConfirmEnd(false)}>Keep going</Button><Button variant="danger" onClick={() => { setConfirmEnd(false); ctl.end(); }}>End and see report</Button></>}
      >
        <p className="text-[14px] text-fg-muted">Your report will cover the questions answered so far.</p>
      </Dialog>

      {(finishing || state.done) && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-canvas/80 backdrop-blur">
          <div className="flex flex-col items-center gap-3 text-center">
            <Spinner className="size-6" />
            <p className="text-[16px] font-medium">Building your report…</p>
            <p className="text-[13px] text-fg-muted">Scoring each answer against your own words. This takes a few seconds.</p>
          </div>
        </div>
      )}

      {debug && state.joined && <DiagPanel ctl={ctl} state={state} />}
    </div>
  );
}

// ----------------------------------------------------------------------------- pieces

function TopBar({ info, state, persona }: { info: SessionInfo; state: RoomState; persona: Persona }) {
  const r = state.remainingS ?? info.params.duration_minutes * 60;
  const low = r <= 90;
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-3 px-4">
        <BrandMark />
        <span className="hidden h-5 w-px bg-line sm:block" />
        <div className="hidden min-w-0 sm:block">
          <p className="truncate text-[13px] font-medium">{info.params.role}{info.params.company ? ` · ${info.params.company}` : ""}</p>
          <p className="truncate text-[12px] text-fg-subtle">{label(info.params.interview_type)} · {label(info.params.round)} · with {persona.name}</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {state.joined && <ConnBadge state={state} />}
          <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-[13px] tabular", low ? "border-warn/40 text-warn" : "border-line text-fg")} aria-label="Time remaining">
            <Clock className="size-3.5" /> {fmt(r)}
          </span>
        </div>
      </div>
    </header>
  );
}

function ConnBadge({ state }: { state: RoomState }) {
  if (state.conn === "open") {
    const map: Record<string, [string, "live" | "accent" | "warn" | "neutral"]> = {
      listening: ["Listening", "live"], speaking: ["Speaking", "accent"], thinking: ["Thinking", "warn"], idle: ["Connected", "neutral"], done: ["Finished", "neutral"],
    };
    const [text, tone] = map[state.phase] ?? ["Connected", "neutral"];
    return <Badge tone={tone} className="gap-1.5"><Wifi className="size-3" /> {text}</Badge>;
  }
  if (state.conn === "failed") return <Badge tone="danger"><WifiOff className="size-3" /> {state.connDetail || "Disconnected"}</Badge>;
  return <Badge tone="warn"><Spinner className="size-3" /> {state.conn === "reconnecting" ? "Reconnecting…" : "Connecting…"}</Badge>;
}

function Captions({ state, persona }: { state: RoomState; persona: string }) {
  const user = [...state.finals, state.partial].filter(Boolean).join(" ");
  const showUser = state.phase === "listening" || (state.phase === "thinking" && user);
  const text = showUser ? user : state.interviewerCaption;
  return (
    <div className="pointer-events-none absolute inset-x-4 bottom-4 flex justify-center sm:right-52" aria-live="polite" data-testid="captions">
      <div className={cn("max-w-[720px] rounded-[14px] border border-line bg-canvas/75 px-4 py-3 backdrop-blur-md transition-opacity", text || state.phase === "listening" ? "opacity-100" : "opacity-0")}>
        <p className="mb-0.5 text-[11px] font-medium uppercase tracking-wide text-fg-subtle">
          {showUser ? (state.muted ? "You (muted)" : "You") : persona}
          {state.phase === "thinking" && <span className="ml-2 normal-case tracking-normal text-warn">thinking…</span>}
        </p>
        <p className="text-[15px] leading-6">
          {text || (state.phase === "listening" ? <span className="text-fg-muted">{state.muted ? "You're muted. Press M to unmute." : "Go ahead, I'm listening…"}</span> : "")}
        </p>
      </div>
    </div>
  );
}

function SidePanel({ info, state, sessionId }: { info: SessionInfo; state: RoomState; sessionId: string }) {
  const box = useRef<HTMLDivElement>(null);
  // keep the latest line in view by scrolling the transcript box only (never the page)
  useEffect(() => { const b = box.current; if (b) b.scrollTo({ top: b.scrollHeight, behavior: "smooth" }); }, [state.transcript.length]);
  const comps = info.blueprint?.competencies ?? [];
  const asked = new Map<string, number>();
  state.transcript.forEach((l) => l.role === "interviewer" && l.competency && asked.set(l.competency, (asked.get(l.competency) ?? 0) + 1));
  return (
    <aside className="flex min-h-0 flex-col gap-4 lg:max-h-[calc(100dvh-160px)]">
      {state.notices.map((n, i) => <Alert key={`${n.event}-${i}`} tone="warn">{n.message}</Alert>)}
      {comps.length > 0 && (
        <Card className="p-4">
          <p className="mb-3 text-[12px] font-medium uppercase tracking-wide text-fg-subtle">Interview plan</p>
          <ul className="flex flex-col gap-2">
            {comps.map((c) => {
              const n = asked.get(c.name) ?? 0;
              const active = state.current?.competency === c.name;
              return (
                <li key={c.id} className="flex items-center gap-2 text-[13px]">
                  <span className={cn("size-2 rounded-full", active ? "bg-accent" : n ? "bg-live" : "bg-surface-3")} />
                  <span className={cn("flex-1 truncate", active ? "font-medium text-fg" : "text-fg-muted")}>{c.name}</span>
                  <span className="font-mono text-[11px] text-fg-subtle">{n} q · {c.minutes}m</span>
                </li>
              );
            })}
          </ul>
        </Card>
      )}
      {info.challenge && <CodePanel sessionId={sessionId} challenge={info.challenge} />}
      <Card className="flex min-h-[200px] flex-1 flex-col overflow-hidden">
        <p className="border-b border-line px-4 py-3 text-[12px] font-medium uppercase tracking-wide text-fg-subtle">Transcript</p>
        <div ref={box} className="flex-1 overflow-y-auto px-4 py-3" data-testid="transcript">
          {state.transcript.length === 0 && <p className="text-[13px] text-fg-muted">The conversation will appear here.</p>}
          <ol className="flex flex-col gap-3">
            {state.transcript.map((l) => (
              <li key={l.id} className={cn("flex flex-col gap-1", l.role === "candidate" && "items-end")}>
                <span className="text-[11px] text-fg-subtle">
                  {l.role === "interviewer" ? (state.persona?.name ?? "Interviewer") : "You"}
                  {l.typed ? " · typed" : ""}
                  {l.action && l.role === "interviewer" && l.action !== "open" ? ` · ${label(l.action)}` : ""}
                </span>
                <p className={cn("max-w-[92%] rounded-[12px] px-3 py-2 text-[13px] leading-5", l.role === "interviewer" ? "bg-surface-2" : "bg-accent/15")}>
                  {l.text}
                  {l.emergency && <span className="mt-1 block text-[11px] text-warn">Backup question (AI unavailable)</span>}
                </p>
              </li>
            ))}
          </ol>
        </div>
      </Card>
    </aside>
  );
}

function Dock({ state, hasMic, onMute, onType, onRepeat, onSkip, onEnd }: {
  state: RoomState; hasMic: boolean; onMute: () => void; onType: () => void; onRepeat: () => void; onSkip: () => void; onEnd: () => void;
}) {
  const busy = state.phase === "thinking" || state.phase === "done" || state.conn !== "open";
  return (
    <div className="fixed inset-x-0 bottom-4 z-30 flex justify-center px-4">
      <div className="flex items-center gap-1.5 rounded-[18px] border border-line bg-surface/90 p-1.5 shadow-[var(--shadow-float)] backdrop-blur" role="toolbar" aria-label="Interview controls">
        <DockButton label={state.muted ? "Unmute (M)" : "Mute (M)"} onClick={onMute} disabled={!hasMic} active={state.muted} danger={state.muted}>
          {state.muted || !hasMic ? <MicOff className="size-5" /> : <Mic className="size-5" />}
        </DockButton>
        <DockButton label="Type an answer (T)" onClick={onType}><Keyboard className="size-5" /></DockButton>
        <DockButton label="Repeat the question" onClick={onRepeat} disabled={busy}><Repeat2 className="size-5" /></DockButton>
        <DockButton label="Skip this question" onClick={onSkip} disabled={busy || state.phase !== "listening"}><SkipForward className="size-5" /></DockButton>
        <span className="mx-1 h-6 w-px bg-line" />
        <button type="button" onClick={onEnd} className="inline-flex h-11 items-center gap-2 rounded-[14px] bg-danger/90 px-4 text-[14px] font-medium text-white hover:bg-danger" data-testid="end">
          <PhoneOff className="size-4" /> <span className="hidden sm:inline">End</span>
        </button>
      </div>
    </div>
  );
}

function DockButton({ label, onClick, disabled, active, danger, children }: { label: string; onClick: () => void; disabled?: boolean; active?: boolean; danger?: boolean; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      aria-pressed={active}
      className={cn("grid size-11 place-items-center rounded-[14px] transition-colors disabled:opacity-40", danger ? "bg-danger/15 text-danger" : "text-fg hover:bg-surface-3")}
    >
      {children}
    </button>
  );
}

function TypeBox({ onSend, onClose, disabled }: { onSend: (t: string) => void; onClose: () => void; disabled: boolean }) {
  const [text, setText] = useState("");
  return (
    <div className="fixed inset-x-0 bottom-24 z-30 flex justify-center px-4">
      <Card className="flex w-[min(96vw,640px)] flex-col gap-2 p-3 shadow-[var(--shadow-float)]">
        <Textarea
          autoFocus
          rows={3}
          value={text}
          placeholder="Type your answer… (Enter to send, Shift+Enter for a new line)"
          aria-label="Type your answer"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") onClose();
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!disabled && text.trim()) { onSend(text); setText(""); } }
          }}
        />
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
          <Button size="sm" disabled={disabled || !text.trim()} onClick={() => { onSend(text); setText(""); }} data-testid="send-text">Send answer</Button>
        </div>
      </Card>
    </div>
  );
}

function FullScreen({ children }: { children: React.ReactNode }) {
  return <div className="grid min-h-dvh place-items-center bg-canvas p-6">{children}</div>;
}

function fmt(s: number) {
  const m = Math.floor(Math.max(0, s) / 60);
  return `${m}:${String(Math.max(0, s) % 60).padStart(2, "0")}`;
}

function label(s: string) {
  if (s === "hr") return "HR";
  return s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function errorTitle(code: string) {
  return (
    { stt_unavailable: "Speech recognition unavailable", stt_failed: "We missed that", tts_failed: "Voice playback problem", turn_failed: "The interviewer hit a problem" } as Record<string, string>
  )[code] ?? "Something went wrong";
}
