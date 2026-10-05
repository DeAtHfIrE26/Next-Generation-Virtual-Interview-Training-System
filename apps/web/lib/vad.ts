// Voice activity detection and turn-taking for the interview room.
// Energy-based with an adaptive noise floor; pure (no browser APIs) so it is unit-tested.
// Barge-in: if the candidate starts speaking while the interviewer is talking, the
// interviewer stops (the caller cancels TTS) and the candidate's turn begins.

export type VadState = "idle" | "speech" | "hangover";
export type VadEvent =
  | { type: "speech_start"; t: number; bargeIn: boolean }
  | { type: "speech_end"; t: number; durationMs: number };

export interface VadOptions {
  frameMs: number; // duration of each frame passed to push()
  onsetMs: number; // speech must persist this long to start
  hangoverMs: number; // silence that ends an utterance
  marginDb: number; // speech threshold above the noise floor
  minSpeechDb: number; // absolute floor (dBFS) to ignore very quiet rooms
  calibrationMs: number; // initial window used only to measure the room's noise floor
}

export const DEFAULT_VAD: VadOptions = { frameMs: 20, onsetMs: 120, hangoverMs: 1200, marginDb: 12, minSpeechDb: -55, calibrationMs: 400 };

export function rmsDb(frame: Float32Array): number {
  let s = 0;
  for (let i = 0; i < frame.length; i++) s += frame[i]! * frame[i]!;
  const rms = Math.sqrt(s / Math.max(frame.length, 1));
  return 20 * Math.log10(rms + 1e-9);
}

export class Vad {
  state: VadState = "idle";
  noiseDb = -60;
  private above = 0;
  private below = 0;
  private startT = 0;
  private calibrated = 0;
  private calibSum = 0;
  constructor(private readonly o: VadOptions = DEFAULT_VAD) {}

  /** Feed one frame's level (dBFS) at time t (ms). `agentSpeaking` marks barge-in. */
  push(db: number, t: number, agentSpeaking = false): VadEvent | null {
    if (this.calibrated < this.o.calibrationMs) {
      // Measure the room before listening for speech, so a noisy room is not "speech".
      this.calibSum += db;
      this.calibrated += this.o.frameMs;
      this.noiseDb = this.calibSum / (this.calibrated / this.o.frameMs);
      return null;
    }
    const threshold = Math.max(this.noiseDb + this.o.marginDb, this.o.minSpeechDb);
    const loud = db > threshold;
    if (!loud && this.state === "idle") {
      // Track the noise floor only while nobody is speaking (slow rise, fast fall).
      this.noiseDb = db < this.noiseDb ? db : this.noiseDb + 0.05 * (db - this.noiseDb);
    }
    if (this.state === "idle") {
      this.above = loud ? this.above + this.o.frameMs : 0;
      if (this.above >= this.o.onsetMs) {
        this.state = "speech";
        this.startT = Math.max(0, t - this.above);
        this.below = 0;
        return { type: "speech_start", t: this.startT, bargeIn: agentSpeaking };
      }
      return null;
    }
    if (loud) {
      this.below = 0;
      this.state = "speech";
      return null;
    }
    this.below += this.o.frameMs;
    this.state = "hangover";
    if (this.below >= this.o.hangoverMs) {
      this.state = "idle";
      this.above = 0;
      const end = t - this.below;
      return { type: "speech_end", t: end, durationMs: end - this.startT };
    }
    return null;
  }
}
