// Interviewer speech (browser speech synthesis or server audio) and browser speech
// recognition for live captions. Both are feature-detected; nothing here is required for the
// session to work (typed answers and captions-only mode remain available).
import { forWord, fromText, type Timeline } from "./visemes";

export interface Speaking { cancel(): void; done: Promise<void> }

export function speakBrowser(text: string, onTimeline: (tl: Timeline) => void, onStart?: () => void): Speaking {
  const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;
  if (!synth) {
    onTimeline(fromText(text, text.split(/\s+/).length * 350));
    return { cancel() {}, done: new Promise((r) => setTimeout(r, text.split(/\s+/).length * 350)) };
  }
  const u = new SpeechSynthesisUtterance(text);
  const voices = synth.getVoices();
  u.voice = voices.find((v) => v.lang === "en-IN") ?? voices.find((v) => v.lang.startsWith("en")) ?? null;
  u.rate = 1.0;
  const tl: Timeline = [];
  let t0 = 0;
  let sawBoundary = false;
  u.onstart = () => {
    t0 = performance.now();
    onStart?.();
    // Fallback timeline until/unless boundary events arrive.
    onTimeline(fromText(text, text.split(/\s+/).length * 330));
  };
  u.onboundary = (e) => {
    if (e.name !== "word") return;
    if (!sawBoundary) { sawBoundary = true; tl.length = 0; }
    const word = text.slice(e.charIndex, e.charIndex + (e.charLength || text.slice(e.charIndex).search(/\s|$/)));
    tl.push(...forWord(word, performance.now() - t0, Math.max(150, word.length * 60)));
    onTimeline([...tl]);
  };
  const done = new Promise<void>((resolve) => { u.onend = () => resolve(); u.onerror = () => resolve(); });
  synth.cancel();
  synth.speak(u);
  return { cancel: () => synth.cancel(), done };
}

export function playServerAudio(b64: string, mime: string, visemes: [number, string][],
  onTimeline: (tl: Timeline) => void, onStart?: () => void): Speaking {
  const audio = new Audio(`data:${mime};base64,${b64}`);
  const done = new Promise<void>((resolve) => { audio.onended = () => resolve(); audio.onerror = () => resolve(); });
  audio.onplay = () => { onStart?.(); onTimeline(visemes as Timeline); };
  void audio.play().catch(() => onTimeline([]));
  return { cancel: () => { audio.pause(); audio.currentTime = 0; }, done };
}

type RecognitionCtor = new () => {
  lang: string; continuous: boolean; interimResults: boolean;
  onresult: ((e: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null;
  onerror: (() => void) | null; onend: (() => void) | null; start(): void; stop(): void;
};

export class Captions {
  private rec: InstanceType<RecognitionCtor> | null = null;
  finalText = "";
  constructor(private onText: (interim: string, final: string) => void) {}

  static supported(): boolean {
    return typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);
  }

  start() {
    const W = window as unknown as { SpeechRecognition?: RecognitionCtor; webkitSpeechRecognition?: RecognitionCtor };
    const Ctor = W.SpeechRecognition ?? W.webkitSpeechRecognition;
    if (!Ctor) return;
    this.finalText = "";
    const rec = new Ctor();
    rec.lang = "en-IN";
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i]!;
        if (r.isFinal) this.finalText += `${r[0]!.transcript} `;
        else interim += r[0]!.transcript;
      }
      this.onText(interim, this.finalText.trim());
    };
    rec.onerror = () => {};
    this.rec = rec;
    rec.start();
  }

  stop(): string {
    this.rec?.stop();
    this.rec = null;
    return this.finalText.trim();
  }
}
