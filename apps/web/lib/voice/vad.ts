// Client-side voice activity detection. Primary: Silero VAD v5 (MIT model) in an AudioWorklet via
// @ricky0123/vad-web, assets served from /vendor. Fallback: the energy VAD in lib/vad.ts, used only
// if the model or WASM runtime cannot load. Used for barge-in and the speaking indicator; the
// server makes the end-of-turn decision from its own VAD + transcript.

import { Vad as EnergyVad, rmsDb } from "../vad";

export interface VadHandlers {
  onSpeechStart: () => void;
  onSpeechEnd: () => void;
  onProbability?: (p: number) => void;
}

export interface ClientVad {
  kind: "silero" | "energy";
  pushFrame?: (pcm16: Int16Array) => void; // energy fallback is fed by the capture worklet
  destroy: () => Promise<void>;
}

export async function startVad(stream: MediaStream, h: VadHandlers): Promise<ClientVad> {
  try {
    const { MicVAD } = await import("@ricky0123/vad-web");
    const vad = await MicVAD.new({
      model: "v5",
      baseAssetPath: "/vendor/vad/",
      onnxWASMBasePath: "/vendor/ort/",
      getStream: async () => stream,
      pauseStream: async () => undefined,
      resumeStream: async () => stream,
      positiveSpeechThreshold: 0.6,
      negativeSpeechThreshold: 0.4,
      minSpeechMs: 250,
      redemptionMs: 600,
      preSpeechPadMs: 100,
      onSpeechStart: () => h.onSpeechStart(),
      onSpeechEnd: () => h.onSpeechEnd(),
      onVADMisfire: () => h.onSpeechEnd(),
      onFrameProcessed: (p) => h.onProbability?.(p.isSpeech),
      startOnLoad: true,
    });
    return { kind: "silero", destroy: () => vad.destroy() };
  } catch (e) {
    console.warn("Silero VAD unavailable, using energy VAD", e);
    const ev = new EnergyVad();
    let t = 0;
    let speaking = false;
    return {
      kind: "energy",
      pushFrame: (pcm16) => {
        const f = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) f[i] = (pcm16[i] ?? 0) / 32768;
        t += 20;
        const db = rmsDb(f);
        const r = ev.push(db, t, false);
        h.onProbability?.(Math.max(0, Math.min(1, (db + 60) / 40)));
        if (r?.type === "speech_start" && !speaking) { speaking = true; h.onSpeechStart(); }
        if (r?.type === "speech_end" && speaking) { speaking = false; h.onSpeechEnd(); }
      },
      destroy: async () => undefined,
    };
  }
}
