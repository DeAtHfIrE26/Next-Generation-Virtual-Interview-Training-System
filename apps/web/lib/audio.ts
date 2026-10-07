// Microphone capture: 20 ms level frames for VAD, plus recording of one utterance as 16 kHz PCM.
import { rmsDb } from "./vad";
import { resample } from "./wav";

export interface Recording { samples: Float32Array; sampleRate: number; startedAt: number }

export class MicCapture {
  private ctx!: AudioContext;
  private node!: AudioWorkletNode;
  private pending: number[] = [];
  private chunks: Float32Array[] = [];
  private recording = false;
  private recStart = 0;
  onFrame: ((db: number, tMs: number) => void) | null = null;
  constructor(readonly stream: MediaStream) {}

  static async open(stream?: MediaStream): Promise<MicCapture> {
    const s = stream ?? (await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    }));
    const m = new MicCapture(s);
    await m.init();
    return m;
  }

  private async init() {
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/worklets/pcm-capture.js");
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture");
    const frame = Math.round(this.ctx.sampleRate * 0.02);
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      if (this.recording) this.chunks.push(e.data);
      for (const v of e.data) this.pending.push(v);
      while (this.pending.length >= frame) {
        const f = Float32Array.from(this.pending.splice(0, frame));
        this.onFrame?.(rmsDb(f), performance.now());
      }
    };
    src.connect(this.node);
  }

  get sampleRate() { return this.ctx.sampleRate; }

  startRecording(): number {
    this.chunks = [];
    this.recording = true;
    this.recStart = performance.now();
    return this.recStart;
  }

  stopRecording(): Recording {
    this.recording = false;
    const n = this.chunks.reduce((s, c) => s + c.length, 0);
    const all = new Float32Array(n);
    let o = 0;
    for (const c of this.chunks) { all.set(c, o); o += c.length; }
    return { samples: resample(all, this.ctx.sampleRate, 16000), sampleRate: 16000, startedAt: this.recStart };
  }

  async close() {
    this.node?.disconnect();
    await this.ctx?.close();
  }
}
