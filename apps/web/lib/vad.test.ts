import { describe, expect, it } from "vitest";
import { Vad, rmsDb, type VadEvent } from "./vad";

function run(levels: number[], agent = false) {
  const v = new Vad();
  const events: VadEvent[] = [];
  levels.forEach((db, i) => { const e = v.push(db, i * 20, agent); if (e) events.push(e); });
  return events;
}

describe("VAD", () => {
  it("detects an utterance with onset and hangover", () => {
    const levels = [...Array(50).fill(-60), ...Array(100).fill(-20), ...Array(80).fill(-60)];
    const ev = run(levels);
    expect(ev.map((e) => e.type)).toEqual(["speech_start", "speech_end"]);
    const end = ev[1] as Extract<VadEvent, { type: "speech_end" }>;
    expect(end.durationMs).toBeGreaterThan(1800);
  });

  it("ignores short clicks and keeps short pauses inside one utterance", () => {
    expect(run([...Array(50).fill(-60), -20, -20, ...Array(50).fill(-60)])).toEqual([]);
    const levels = [...Array(50).fill(-60), ...Array(50).fill(-20), ...Array(25).fill(-60), ...Array(50).fill(-20), ...Array(80).fill(-60)];
    expect(run(levels).filter((e) => e.type === "speech_start")).toHaveLength(1);
  });

  it("flags barge-in when the agent is speaking", () => {
    const ev = run([...Array(20).fill(-60), ...Array(20).fill(-15)], true);
    expect(ev[0]).toMatchObject({ type: "speech_start", bargeIn: true });
  });

  it("adapts to a noisy room", () => {
    const ev = run([...Array(200).fill(-35), ...Array(30).fill(-30), ...Array(100).fill(-35)]);
    expect(ev).toEqual([]); // +5 dB over a -35 dB floor is not speech
  });

  it("computes dBFS", () => {
    expect(rmsDb(new Float32Array(160).fill(1))).toBeCloseTo(0, 3);
    expect(rmsDb(new Float32Array(160))).toBeLessThan(-150);
  });
});
