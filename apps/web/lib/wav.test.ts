import { describe, expect, it } from "vitest";
import { encodeWav, resample, toBase64 } from "./wav";

describe("wav", () => {
  it("encodes a valid 16-bit mono header", () => {
    const b = encodeWav(new Float32Array([0, 1, -1]), 16000);
    const v = new DataView(b.buffer);
    expect(String.fromCharCode(...b.subarray(0, 4))).toBe("RIFF");
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getInt16(46, true)).toBe(32767);
    expect(v.getInt16(48, true)).toBe(-32768);
    expect(b.length).toBe(44 + 6);
  });
  it("resamples by duration", () => {
    expect(resample(new Float32Array(48000), 48000, 16000).length).toBe(16000);
    const r = resample(new Float32Array([0, 1]), 2, 4);
    expect(r[0]).toBe(0);
    expect(r[r.length - 1]).toBe(1);
  });
  it("base64 round-trips", () => {
    expect(atob(toBase64(new Uint8Array([104, 105])))).toBe("hi");
  });
});
