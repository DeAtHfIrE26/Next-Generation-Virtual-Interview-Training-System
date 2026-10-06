// Realtime interview connection (protocol: services/api/src/interview_api/live.py).
// One WebSocket per session. Reconnects automatically with backoff and a fresh ticket, so a network
// blip never loses the session: the server keeps state and re-asks the pending question on "start".

import { api, ApiError } from "../api";

export type ServerMessage =
  | { type: "ready"; session: SessionSnapshot; providers: Providers }
  | { type: "phase"; phase: Phase }
  | { type: "stt.partial"; text: string }
  | { type: "stt.final"; text: string; segment: number; final?: boolean }
  | { type: "turn.end"; turn: number }
  | { type: "thinking" }
  | { type: "question"; utterance: number } & PublicTurn
  | { type: "tts.start"; utterance: number; sample_rate: number; provider: string }
  | { type: "tts.chunk"; utterance: number; text: string; offset: number; duration: number; marks: Mark[] }
  | { type: "tts.end"; utterance: number; audio: boolean; duration?: number }
  | { type: "tts.cancel"; utterance: number }
  | { type: "listening"; turn: number }
  | { type: "notice"; event: string; message: string; episode: number; end_session: boolean }
  | { type: "diag"; stage: string; ms: number; provider?: string }
  | { type: "error"; code: string; message?: string; recoverable: boolean }
  | { type: "done"; session_id: string }
  | { type: "pong"; t: number };

export type Phase = "idle" | "speaking" | "listening" | "thinking" | "done";
export interface Mark { t: number; kind: string; value: string }
export interface Providers { llm: string | null; llm_model: string | null; llm_fallback: string | null; stt: string | null; tts: string | null }
export interface PublicTurn {
  index: number; action: string; competency: string; difficulty: number; say: string; emergency: boolean; answered: boolean; answer: string | null;
}
export interface SessionSnapshot {
  status: string; finished: boolean; turns: PublicTurn[]; current: PublicTurn | null;
  persona: Persona; language: string; remaining_s: number;
}
export interface Persona { id: string; name: string; title: string; kokoro_voice: string; polly_voice: string; look: Record<string, string> }

export type ConnState = "connecting" | "open" | "reconnecting" | "closed" | "failed";

export interface RealtimeHandlers {
  onMessage: (m: ServerMessage) => void;
  onAudio: (utterance: number, pcm: ArrayBuffer) => void;
  onState: (s: ConnState, detail?: string) => void;
  onRtt?: (ms: number) => void;
}

export class RealtimeClient {
  private ws: WebSocket | null = null;
  private attempts = 0;
  private stopped = false;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly onlineHandler = () => { if (!this.ws || this.ws.readyState > 1) this.reconnectNow(); };
  everOpened = false;
  reconnects = 0;

  constructor(private sessionId: string, private h: RealtimeHandlers) {
    window.addEventListener("online", this.onlineHandler);
  }

  async connect(): Promise<void> {
    if (this.stopped) return;
    this.h.onState(this.everOpened ? "reconnecting" : "connecting");
    let ticket: { ticket: string; url: string };
    try {
      ticket = await api(`/sessions/${this.sessionId}/realtime`, { method: "POST" });
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 404 || e.status === 409)) {
        this.h.onState("failed", e.message);
        return;
      }
      return this.scheduleRetry();
    }
    const ws = new WebSocket(`${ticket.url}?ticket=${encodeURIComponent(ticket.ticket)}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this.attempts = 0;
      if (this.everOpened) this.reconnects += 1;
      this.everOpened = true;
      this.h.onState("open");
      this.pingTimer = setInterval(() => this.send({ type: "ping", t: performance.now() }), 10_000);
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const m = JSON.parse(ev.data) as ServerMessage;
        if (m.type === "pong") this.h.onRtt?.(performance.now() - m.t);
        this.h.onMessage(m);
      } else {
        const buf = ev.data as ArrayBuffer;
        const utt = new Uint8Array(buf, 0, 1)[0] ?? 0;
        this.h.onAudio(utt, buf.slice(1));
      }
    };
    ws.onclose = (ev) => {
      if (this.pingTimer) clearInterval(this.pingTimer);
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.stopped || ev.code === 1000) return this.h.onState("closed");
      if (ev.code === 4409) return this.h.onState("failed", "This interview was opened in another tab.");
      this.scheduleRetry();
    };
    ws.onerror = () => undefined; // onclose follows and handles it
  }

  private scheduleRetry() {
    if (this.stopped) return;
    this.attempts += 1;
    if (this.attempts > 8) return this.h.onState("failed", "We couldn't reconnect. Check your connection and reload.");
    this.h.onState("reconnecting");
    const delay = Math.min(8000, 400 * 2 ** (this.attempts - 1)) + Math.random() * 250; // jitter avoids thundering herd
    this.retryTimer = setTimeout(() => void this.connect(), delay);
  }

  private reconnectNow() {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.attempts = 0;
    void this.connect();
  }

  get open() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  send(obj: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
  }

  sendAudio(pcm: ArrayBuffer) {
    // Drop audio while (re)connecting rather than buffering stale speech.
    if (this.ws?.readyState === WebSocket.OPEN && this.ws.bufferedAmount < 256_000) this.ws.send(pcm);
  }

  close() {
    this.stopped = true;
    window.removeEventListener("online", this.onlineHandler);
    if (this.retryTimer) clearTimeout(this.retryTimer);
    if (this.pingTimer) clearInterval(this.pingTimer);
    this.ws?.close(1000, "bye");
    this.ws = null;
  }
}
