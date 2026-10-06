// The live interview controller: wires the microphone, client VAD, realtime connection, the
// interviewer's voice (Speaker) and on-device vision together, and exposes one immutable state
// snapshot for React (useSyncExternalStore). Kept outside React so audio callbacks never see stale
// closures.

import { api } from "../api";
import type { AvatarState, Speaker } from "../avatar/speaker";
import { MicCapture } from "../voice/mic";
import { RealtimeClient, type ConnState, type Persona, type Phase, type Providers, type PublicTurn, type ServerMessage } from "../voice/realtime";
import { startVad, type ClientVad } from "../voice/vad";
import { faceCrop, type VisionSample } from "../vision";

export interface Notice { event: string; message: string; end_session?: boolean }
export interface Line { id: number; role: "interviewer" | "candidate"; text: string; emergency?: boolean; competency?: string; action?: string; typed?: boolean }

export interface RoomState {
  conn: ConnState;
  connDetail: string;
  phase: Phase;
  joined: boolean;
  providers: Providers | null;
  persona: Persona | null;
  current: PublicTurn | null;
  interviewerCaption: string;
  partial: string;
  finals: string[];
  transcript: Line[];
  remainingS: number | null;
  notices: Notice[];
  error: { code: string; message: string } | null;
  muted: boolean;
  userSpeaking: boolean;
  vadKind: "silero" | "energy" | "none";
  done: boolean;
  emergencyCount: number;
  audioPlaying: boolean;
}

export interface DiagSnapshot {
  rttMs: number;
  stages: Record<string, number[]>;
  reconnects: number;
  bufferedAudio: number;
  speakerKind: string;
  fps: number;
  vadProb: number;
  micLevel: number;
  sampleRate: number;
  audioCtx: string;
  bargeIns: number;
  framesSent: number;
  events: string[]; // recent protocol events (debug panel)
}

const initial: RoomState = {
  conn: "closed", connDetail: "", phase: "idle", joined: false, providers: null, persona: null, current: null,
  interviewerCaption: "", partial: "", finals: [], transcript: [], remainingS: null, notices: [], error: null,
  muted: false, userSpeaking: false, vadKind: "none", done: false, emergencyCount: 0, audioPlaying: false,
};

export class RoomController {
  private state: RoomState = initial;
  private listeners = new Set<() => void>();
  private rt: RealtimeClient | null = null;
  private mic: MicCapture | null = null;
  private vad: ClientVad | null = null;
  private speaker: Speaker | null = null;
  private stream: MediaStream | null = null;
  private lineId = 0;
  // interviewer audio bookkeeping
  private playingUtt = -1;
  private audioUtt = -1;
  private speakerStarted = false;
  private captionChunks: { text: string; offset: number }[] = [];
  private captionTimer: ReturnType<typeof setInterval> | null = null;
  private speakStartedAt = 0;
  // E4/E5 series for the answer in progress (seconds since listening started)
  private listenT0 = 0;
  private mouth = { times: [] as number[], values: [] as (number | null)[] };
  private gaze: [number, boolean | null][] = [];
  private obs: { t: number; type: string; present: boolean }[] = [];
  private lastLandmarks: VisionSample["landmarks"] = null;
  private timers: ReturnType<typeof setInterval>[] = [];
  private countdownAt = 0;
  readonly diag: DiagSnapshot = {
    rttMs: 0, stages: {}, reconnects: 0, bufferedAudio: 0, speakerKind: "-", fps: 0, vadProb: 0, micLevel: 0,
    sampleRate: 0, audioCtx: "-", bargeIns: 0, framesSent: 0, events: [],
  };
  allowBargeIn = true;

  setFaceVerification(on: boolean) { this.opts.faceVerification = on; }

  constructor(readonly sessionId: string, private opts: { faceVerification: boolean; onDone: () => void }) {}

  // ---------------------------------------------------------------- store
  subscribe = (cb: () => void) => {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  };
  getSnapshot = () => this.state;
  private set(patch: Partial<RoomState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((l) => l());
  }
  private push(line: Omit<Line, "id">) {
    this.set({ transcript: [...this.state.transcript, { ...line, id: ++this.lineId }] });
  }

  avatarState(): AvatarState {
    const p = this.state.phase;
    if (p === "speaking") return "speaking";
    if (p === "thinking") return "thinking";
    if (p === "listening") return "listening";
    return "idle";
  }

  // ---------------------------------------------------------------- setup
  attachSpeaker(sp: Speaker) {
    this.speaker = sp;
    this.diag.speakerKind = sp.kind;
    sp.onStarted = () => { this.log("audio started"); this.speakerStarted = true; this.speakStartedAt = performance.now(); this.set({ audioPlaying: true }); this.startCaptionClock(); };
    sp.onEnded = () => this.onPlaybackEnded();
  }

  /** Call from the Join button's click handler (a user gesture: unlocks audio playback). */
  async join(stream: MediaStream | null) {
    if (this.state.joined) return;
    this.set({ joined: true, error: null });
    // resume() can stay pending forever without an audio output device (e.g. headless Firefox):
    // never block joining on it.
    await withTimeout(this.speaker?.unlock() ?? Promise.resolve(), 1500);
    this.stream = stream;
    if (stream && stream.getAudioTracks().length) {
      const audioOnly = new MediaStream(stream.getAudioTracks());
      try {
        this.mic = await MicCapture.start(audioOnly, (pcm) => this.onMicFrame(pcm));
        this.diag.sampleRate = this.mic.ctx.sampleRate;
      } catch (e) {
        console.warn("mic capture failed", e);
      }
      try {
        this.vad = await startVad(audioOnly, {
          onSpeechStart: () => this.onSpeechStart(),
          onSpeechEnd: () => this.set({ userSpeaking: false }),
          onProbability: (p) => { this.diag.vadProb = p; },
        });
        this.set({ vadKind: this.vad.kind });
      } catch (e) {
        console.warn("vad failed", e);
      }
    }
    this.rt = new RealtimeClient(this.sessionId, {
      onMessage: (m) => this.onMessage(m),
      onAudio: (u, pcm) => this.onAudio(u, pcm),
      onState: (s, d) => {
        this.set({ conn: s, connDetail: d ?? "" });
        if (s === "open") this.rt?.send({ type: "start" }); // start, or resume after a reconnect
      },
      onRtt: (ms) => { this.diag.rttMs = ms; },
    });
    await this.rt.connect();
    this.timers.push(setInterval(() => this.flushObservations(), 2000));
    if (this.opts.faceVerification) this.timers.push(setInterval(() => void this.faceCheck(), 45_000));
    this.timers.push(setInterval(() => this.tickDiag(), 500));
  }

  // ---------------------------------------------------------------- mic + VAD
  private loudFrames = 0;

  private onMicFrame(pcm: ArrayBuffer) {
    this.diag.micLevel = this.mic?.level ?? 0;
    // Energy fast path for barge-in: 400 ms of close-mic speech level (about -32 dBFS and up) while
    // the interviewer is audible. Browser echo cancellation keeps speaker leakage well below this;
    // the Silero VAD remains the primary detector, this only covers moments when its inference lags.
    this.loudFrames = this.diag.micLevel > 0.55 ? this.loudFrames + 1 : 0;
    if (this.loudFrames === 20) this.onSpeechStart();
    if (this.vad?.pushFrame) this.vad.pushFrame(new Int16Array(pcm));
    if (this.state.muted) return;
    this.rt?.sendAudio(pcm);
    this.diag.framesSent += 1;
  }

  private onSpeechStart() {
    if (this.state.muted) return;
    this.set({ userSpeaking: true });
    // Barge-in: the candidate talks over the interviewer. Only once audio is actually playing, so a
    // cough during "thinking" doesn't cancel the next question before it starts.
    if (this.allowBargeIn && this.state.phase === "speaking" && this.speakerStarted && !this.isClosingTurn()) {
      this.diag.bargeIns += 1;
      this.log("barge-in");
      this.speaker?.interrupt();
      this.stopCaptionClock(true);
      this.rt?.send({ type: "barge_in" });
    }
  }

  private isClosingTurn() {
    return this.state.current?.action === "close";
  }

  setMuted(m: boolean) {
    this.mic?.setMuted(m);
    this.set({ muted: m, userSpeaking: false });
  }

  // ---------------------------------------------------------------- server messages
  private log(e: string) {
    this.diag.events.push(`${(performance.now() / 1000).toFixed(1)} ${e}`);
    if (this.diag.events.length > 60) this.diag.events.shift();
  }

  private onMessage(m: ServerMessage) {
    if (m.type !== "stt.partial" && m.type !== "pong" && m.type !== "tts.chunk") this.log(`${m.type}${"utterance" in m ? ` u${m.utterance}` : ""}${m.type === "phase" ? ` ${m.phase}` : ""}${m.type === "tts.end" ? ` ${m.duration ?? 0}s` : ""}`);
    switch (m.type) {
      case "ready":
        this.set({ providers: m.providers, persona: m.session.persona, remainingS: Math.round(m.session.remaining_s) });
        if (!this.state.transcript.length) this.seedTranscript(m.session.turns);
        break;
      case "phase":
        this.set({ phase: m.phase });
        break;
      case "question": {
        const { type: _t, utterance, ...turn } = m;
        void _t;
        this.playingUtt = utterance;
        this.speakerStarted = false;
        this.captionChunks = [];
        this.stopCaptionClock(false);
        const last = this.state.transcript.at(-1);
        const isRepeat = last?.role === "interviewer" && last.text === turn.say;
        this.set({
          current: turn, interviewerCaption: "", partial: "", finals: [],
          emergencyCount: this.state.emergencyCount + (turn.emergency && !isRepeat ? 1 : 0),
        });
        if (!isRepeat) this.push({ role: "interviewer", text: turn.say, emergency: turn.emergency, competency: turn.competency, action: turn.action });
        if (this.countdownAt === 0) this.startCountdown();
        break;
      }
      case "tts.start":
        if (m.utterance !== this.playingUtt) break;
        this.audioUtt = m.utterance;
        this.speaker?.start(m.sample_rate);
        break;
      case "tts.chunk":
        if (m.utterance === this.playingUtt) this.captionChunks.push({ text: m.text, offset: m.offset });
        break;
      case "tts.end":
        if (m.utterance !== this.playingUtt) break;
        if (m.audio) this.speaker?.end();
        else this.set({ interviewerCaption: this.state.current?.say ?? "" }); // no audio: server times the turn
        break;
      case "tts.cancel":
        if (m.utterance === this.audioUtt) { this.speaker?.interrupt(); this.stopCaptionClock(true); }
        break;
      case "barge_in": // detected by the server's VAD (the browser's may lag on a busy device)
        this.diag.bargeIns += 1;
        this.speaker?.interrupt();
        this.stopCaptionClock(true);
        break;
      case "listening":
        this.listenT0 = performance.now();
        this.mouth = { times: [], values: [] };
        this.gaze = [];
        this.set({ partial: "", finals: [], interviewerCaption: this.state.current?.say ?? "" });
        break;
      case "stt.partial":
        this.set({ partial: m.text });
        break;
      case "stt.final":
        if (m.final) {
          if (m.text.trim()) this.push({ role: "candidate", text: m.text.trim() });
          this.set({ partial: "", finals: [] });
        } else if (m.text.trim()) {
          this.set({ finals: [...this.state.finals, m.text.trim()], partial: "" });
        }
        break;
      case "turn.end":
        // E4 (lip-sync) and E5 (gaze) series for the answer that just ended.
        this.rt?.send({ type: "answer_meta", mouth: this.mouth.times.length > 10 ? this.mouth : null, gaze: this.gaze.length > 10 ? this.gaze : null });
        break;
      case "notice":
        this.set({ notices: [{ event: m.event, message: m.message, end_session: m.end_session }, ...this.state.notices].slice(0, 4) });
        break;
      case "diag":
        (this.diag.stages[m.stage] ??= []).push(m.ms);
        break;
      case "error":
        this.set({ error: { code: m.code, message: m.message ?? m.code } });
        break;
      case "done":
        this.set({ done: true, phase: "done" });
        this.opts.onDone();
        break;
      default:
        break;
    }
  }

  private seedTranscript(turns: PublicTurn[]) {
    for (const t of turns) {
      this.push({ role: "interviewer", text: t.say, emergency: t.emergency, competency: t.competency, action: t.action });
      if (t.answer) this.push({ role: "candidate", text: t.answer });
    }
  }

  private onAudio(uttByte: number, pcm: ArrayBuffer) {
    if (uttByte !== (this.audioUtt & 0xff) || this.audioUtt !== this.playingUtt) return; // stale (cancelled) audio
    this.speaker?.push(pcm);
  }

  private onPlaybackEnded() {
    this.log(`audio ended (playing u${this.playingUtt}, audio u${this.audioUtt})`);
    this.set({ audioPlaying: false });
    this.stopCaptionClock(true);
    if (this.audioUtt === this.playingUtt) this.rt?.send({ type: "playback", state: "ended", utterance: this.playingUtt });
  }

  // Captions follow the audio: each TTS sentence appears when its audio starts playing.
  private startCaptionClock() {
    this.stopCaptionClock(false);
    this.captionTimer = setInterval(() => {
      const t = (performance.now() - this.speakStartedAt) / 1000;
      const text = this.captionChunks.filter((c) => c.offset <= t + 0.05).map((c) => c.text).join(" ");
      if (text !== this.state.interviewerCaption) this.set({ interviewerCaption: text });
    }, 100);
  }
  private stopCaptionClock(showAll: boolean) {
    if (showAll && this.state.audioPlaying) this.set({ audioPlaying: false });
    if (this.captionTimer) clearInterval(this.captionTimer);
    this.captionTimer = null;
    if (showAll && this.state.current) this.set({ interviewerCaption: this.state.current.say });
  }

  private startCountdown() {
    this.countdownAt = performance.now();
    this.timers.push(setInterval(() => {
      const r = this.state.remainingS;
      if (r !== null && r > 0 && this.state.phase !== "done") this.set({ remainingS: r - 1 });
    }, 1000));
  }

  // ---------------------------------------------------------------- vision (E2, E4, E5, E7)
  onVision(s: VisionSample) {
    this.lastLandmarks = s.landmarks;
    const t = s.t / 1000;
    this.obs.push({ t, type: "no_face", present: s.faces === 0 }, { t, type: "second_person", present: s.faces > 1 });
    if (s.phone !== null) this.obs.push({ t, type: "phone", present: s.phone });
    if (this.state.phase === "listening") {
      const rel = (s.t - this.listenT0) / 1000;
      this.mouth.times.push(rel);
      this.mouth.values.push(s.aperture);
      this.gaze.push([rel, s.onScreen]);
    }
  }

  private async flushObservations() {
    if (!this.obs.length || this.state.done) return;
    const batch = this.obs.splice(0, this.obs.length).slice(-500);
    try {
      const r = await api<{ notices: Notice[]; status: string }>(`/sessions/${this.sessionId}/events`, { method: "POST", json: { observations: batch } });
      if (r.notices.length) this.set({ notices: [...r.notices, ...this.state.notices].slice(0, 4) });
      if (r.status === "ended_by_policy") this.end();
    } catch {
      /* transient: the next batch retries */
    }
  }

  private video: HTMLVideoElement | null = null;
  setVideo(v: HTMLVideoElement | null) { this.video = v; }

  private async faceCheck() {
    const lm = this.lastLandmarks;
    if (!lm || !this.video || this.state.done) return;
    const img = faceCrop(this.video, lm);
    if (!img) return;
    const r = await api<{ notice: Notice | null }>(`/sessions/${this.sessionId}/face-check`, { method: "POST", json: { image: img, t: performance.now() / 1000 } }).catch(() => null);
    if (r?.notice) this.set({ notices: [r.notice, ...this.state.notices].slice(0, 4) });
  }

  // ---------------------------------------------------------------- controls
  sendText(text: string) {
    const t = text.trim();
    if (!t) return;
    this.speaker?.interrupt();
    this.stopCaptionClock(true);
    this.push({ role: "candidate", text: t, typed: true });
    this.rt?.send({ type: "text_answer", text: t });
  }
  repeat() { this.speaker?.interrupt(); this.rt?.send({ type: "repeat" }); }
  skip() { this.rt?.send({ type: "skip" }); }
  end() { this.speaker?.interrupt(); this.rt?.send({ type: "end" }); }
  dismissError() { this.set({ error: null }); }

  private tickDiag() {
    this.diag.fps = this.speaker?.fps() ?? 0;
    this.diag.reconnects = this.rt?.reconnects ?? 0;
    this.diag.audioCtx = this.mic?.ctx.state ?? "-";
  }

  async destroy() {
    this.timers.forEach(clearInterval);
    this.stopCaptionClock(false);
    this.rt?.close();
    await this.vad?.destroy().catch(() => undefined);
    await this.mic?.stop();
    this.stream = null;
  }
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T | undefined> {
  return Promise.race([p, new Promise<undefined>((r) => setTimeout(() => r(undefined), ms))]);
}

export function percentile(xs: number[], p: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))] ?? 0;
}
