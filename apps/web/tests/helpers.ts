// Shared E2E helpers.
//
// Fake media: getUserMedia is replaced (before any page script runs) with a stream from a Web Audio
// destination plus a canvas camera. The test then "speaks" real recorded WAV files into it when the
// interviewer is listening. The audio goes through the app's real path: AudioWorklet -> 16 kHz
// PCM16 -> WebSocket -> server VAD + speech recognition. This works on Chromium, Firefox and WebKit
// (Chrome's --use-fake-device flags do not exist elsewhere and cannot be timed to the conversation).
import { expect, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SPEECH_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures/speech");

export async function installFakeMedia(page: Page) {
  await page.addInitScript(() => {
    type FakeMic = { ctx: AudioContext; dest: MediaStreamAudioDestinationNode; playing: number; play: (b64: string) => Promise<number> };
    const w = window as unknown as { __fakeMic?: FakeMic };
    const ensure = (): FakeMic => {
      if (w.__fakeMic) return w.__fakeMic;
      const ctx = new AudioContext();
      const dest = ctx.createMediaStreamDestination();
      // faint noise floor so energy-based detectors have a baseline, like a real room
      const noise = ctx.createBufferSource();
      const nb = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
      const ch = nb.getChannelData(0);
      for (let i = 0; i < ch.length; i++) ch[i] = (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.002;
      noise.buffer = nb;
      noise.loop = true;
      noise.connect(dest);
      noise.start();
      const mic: FakeMic = {
        ctx, dest, playing: 0,
        async play(b64: string) {
          await ctx.resume();
          const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
          const buf = await ctx.decodeAudioData(bytes.buffer);
          const src = ctx.createBufferSource();
          src.buffer = buf;
          src.connect(dest);
          mic.playing += 1;
          src.onended = () => { mic.playing -= 1; };
          src.start();
          return buf.duration;
        },
      };
      w.__fakeMic = mic;
      return mic;
    };
    const fakeCamera = () => {
      const c = document.createElement("canvas");
      c.width = 640;
      c.height = 480;
      const g = c.getContext("2d")!;
      let f = 0;
      setInterval(() => {
        f += 1;
        g.fillStyle = "#223";
        g.fillRect(0, 0, 640, 480);
        g.fillStyle = "#ccb";
        g.beginPath();
        g.ellipse(320 + Math.sin(f / 20) * 6, 230, 90, 120, 0, 0, Math.PI * 2);
        g.fill();
      }, 100);
      return (c as HTMLCanvasElement & { captureStream(fps: number): MediaStream }).captureStream(10).getVideoTracks();
    };
    const md = navigator.mediaDevices;
    if (!md) return;
    md.getUserMedia = async (c?: MediaStreamConstraints) => {
      const mic = ensure();
      const tracks: MediaStreamTrack[] = [];
      if (c?.audio) tracks.push(...mic.dest.stream.getAudioTracks().map((t) => t.clone()));
      if (c?.video) tracks.push(...fakeCamera());
      return new MediaStream(tracks);
    };
    md.enumerateDevices = async () => [
      { deviceId: "fake-mic", kind: "audioinput", label: "Fake microphone (test)", groupId: "g", toJSON() { return this; } },
      { deviceId: "fake-cam", kind: "videoinput", label: "Fake camera (test)", groupId: "g", toJSON() { return this; } },
    ] as MediaDeviceInfo[];
  });
}

/** Plays a fixture WAV into the fake microphone; resolves with its duration in seconds. */
export async function speak(page: Page, file: string): Promise<number> {
  const b64 = fs.readFileSync(path.join(SPEECH_DIR, file)).toString("base64");
  return page.evaluate((b) => (window as unknown as { __fakeMic: { play(b: string): Promise<number> } }).__fakeMic.play(b), b64);
}

/** Registers, grants data-processing consent and creates a session through the web proxy. */
export async function newSession(page: Page, form: Record<string, string>): Promise<string> {
  const email = `e2e-${Date.now()}-${Math.floor(performance.now())}@example.com`;
  const h = { "x-ic-csrf": "1" };
  const reg = await page.request.post("/api/auth/register", {
    headers: h, data: { email, password: "correct horse battery", name: "E2E Candidate", accept_terms: true, age_confirmed: true },
  });
  expect(reg.status(), await reg.text()).toBe(201);
  const consent = await page.request.post("/api/consent", { headers: h, data: { kind: "data_processing", granted: true } });
  expect(consent.ok(), await consent.text()).toBeTruthy();
  const res = await page.request.post("/api/sessions", { headers: h, multipart: form });
  expect(res.status(), await res.text()).toBe(201);
  return ((await res.json()) as { id: string }).id;
}
