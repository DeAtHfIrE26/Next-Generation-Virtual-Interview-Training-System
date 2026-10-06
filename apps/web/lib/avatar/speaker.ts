// The interviewer's voice output. The room only talks to the Speaker interface; it is backed by the
// 3D TalkingHead avatar (streaming playback + audio-driven lip-sync) or, when WebGL is unavailable,
// by a plain Web Audio player. Either way the audio played is exactly the server's TTS audio.

export type AvatarState = "idle" | "listening" | "thinking" | "speaking";

export interface Speaker {
  kind: "talkinghead" | "audio";
  start(sampleRate: number): void;
  push(pcm16: ArrayBuffer): void;
  end(): void;
  interrupt(): void;
  setState(s: AvatarState): void;
  level(): number; // 0..1 output level (drives the speaking ring)
  fps(): number;
  onStarted: (() => void) | null;
  onEnded: (() => void) | null;
  unlock(): Promise<void>; // resume the AudioContext from a user gesture
  destroy(): void;
}

// ----------------------------------------------------------------------------- quality tiers

export interface Quality { tier: "high" | "medium" | "low" | "software"; dpr: number; fps: number }

export function detectQuality(): Quality {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const cores = nav.hardwareConcurrency ?? 4;
  const mem = nav.deviceMemory ?? 4;
  const coarse = matchMedia("(pointer: coarse)").matches;
  let renderer = "";
  try {
    const gl = document.createElement("canvas").getContext("webgl2") as WebGL2RenderingContext | null;
    const ext = gl?.getExtension("WEBGL_debug_renderer_info");
    renderer = (ext && gl ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "").toLowerCase();
  } catch {
    /* no WebGL info */
  }
  const software = /swiftshader|llvmpipe|software/.test(renderer);
  const dpr = window.devicePixelRatio || 1;
  // Software WebGL (VMs, headless CI) renders on the CPU and competes with audio, VAD and vision
  // on the main thread: keep the avatar alive but cheap.
  if (software) return { tier: "software", dpr: 0.5 / dpr, fps: 10 };
  if (cores <= 2 || mem <= 2) return { tier: "low", dpr: 0.75 / dpr, fps: 30 };
  if (coarse || cores <= 4 || mem <= 4) return { tier: "medium", dpr: Math.min(1.5, dpr) / dpr, fps: 45 };
  return { tier: "high", dpr: Math.min(2, dpr) / dpr, fps: 60 };
}

export function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

// ----------------------------------------------------------------------------- TalkingHead (3D)

interface TalkingHeadLike {
  audioCtx: AudioContext;
  audioStreamGainNode: GainNode;
  audioAnalyzerNode: AnalyserNode;
  renderer?: { setPixelRatio(r: number): void };
  mtAvatar: Record<string, { newvalue: number; needsUpdate: boolean }>;
  opt: { update: ((dt: number) => void) | null; modelFPS: number };
  armature?: { traverse(fn: (o: { isMesh?: boolean; material?: { name?: string; color?: { set(c: string): void } } }) => void): void };
  showAvatar(avatar: Record<string, unknown>, onprogress?: (e: ProgressEvent) => void): Promise<void>;
  streamStart(opt: Record<string, unknown>, onStart?: () => void, onEnd?: () => void): Promise<void> | void;
  streamAudio(r: { audio: ArrayBuffer }): void;
  streamNotifyEnd(): void;
  streamInterrupt(): void;
  streamStop(): void;
  setMood(m: string): void;
  makeEyeContact(t: number): void;
  lookAhead(t: number): void;
  lookAt(x: number | null, y: number | null, t: number): void;
  speakWithHands(delay?: number, prob?: number): void;
  stop(): void;
  start(): void;
}

export interface Look { skin?: string; hair?: string; top?: string }

export async function createTalkingHeadSpeaker(
  el: HTMLElement,
  look: Look,
  quality: Quality,
  onProgress: (pct: number) => void,
): Promise<Speaker> {
  const thUrl = "/vendor/talkinghead/talkinghead.mjs";
  const haUrl = "/vendor/headaudio/headaudio.min.mjs";
  const { TalkingHead } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ thUrl);
  const head: TalkingHeadLike = new TalkingHead(el, {
    ttsEndpoint: null,
    lipsyncModules: ["en"],
    lipsyncLang: "en",
    cameraView: "upper",
    cameraDistance: 0.25,
    cameraY: 0.04,
    cameraRotateEnable: false,
    cameraPanEnable: false,
    cameraZoomEnable: false,
    modelFPS: quality.fps,
    modelPixelRatio: quality.dpr,
    modelMovementFactor: 0.8,
    avatarIdleEyeContact: 0.55,
    avatarIdleHeadMove: 0.45,
    avatarSpeakingEyeContact: 0.75,
    avatarSpeakingHeadMove: 0.5,
    lightAmbientIntensity: 1.6,
    lightDirectIntensity: 26,
    lightDirectColor: 0xbfc4ff,
    lightSpotIntensity: 6,
    lightSpotColor: 0x55e0d0,
    mixerGainSpeech: 2,
  });
  await head.showAvatar(
    { url: "/avatars/interviewer.glb", body: "F", avatarMood: "neutral", lipsyncLang: "en" },
    (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100)),
  );
  applyLook(head, look);

  // Audio-driven lip-sync: HeadAudio listens to the interviewer audio actually playing.
  await head.audioCtx.audioWorklet.addModule("/vendor/headaudio/headworklet.min.mjs");
  const { HeadAudio } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ haUrl);
  const ha = new HeadAudio(head.audioCtx, { processorOptions: {}, parameterData: { vadGateActiveDb: -45, vadGateInactiveDb: -60, speakerMeanHz: 210 } });
  await ha.loadModel("/vendor/headaudio/model-en-mixed.bin");
  head.audioStreamGainNode.connect(ha);
  ha.onvalue = (key: string, value: number) => {
    const t = head.mtAvatar[key];
    if (t) Object.assign(t, { newvalue: value, needsUpdate: true });
  };
  let frames = 0;
  let last = performance.now();
  let measured = quality.fps;
  head.opt.update = (dt: number) => {
    ha.update(dt);
    frames += 1;
    const now = performance.now();
    if (now - last >= 1000) {
      measured = (frames * 1000) / (now - last);
      frames = 0;
      last = now;
    }
  };

  const analyser = head.audioAnalyzerNode;
  const buf = new Uint8Array(analyser.fftSize);
  let streaming = false;
  let sr = 0;
  // Per-utterance bookkeeping. TalkingHead's stream callbacks fire once per stream session, not per
  // utterance, so start/end are tracked here from the audio actually queued.
  let utt = 0;
  let started = false;
  let ended = true;
  let endRequested = false; // tts.end received: no more audio is coming for this utterance
  let firstAt = 0;
  let queuedS = 0;
  let endTimer: ReturnType<typeof setTimeout> | null = null;
  let nod: ReturnType<typeof setInterval> | null = null;
  const finish = (u: number) => {
    if (u !== utt || ended) return;
    ended = true;
    if (endTimer) clearTimeout(endTimer);
    endTimer = null;
    sp.onEnded?.();
  };
  const sp: Speaker = {
    kind: "talkinghead",
    onStarted: null,
    onEnded: null,
    start(sampleRate) {
      if (!streaming || sr !== sampleRate) {
        void head.streamStart(
          { sampleRate, lipsyncType: "visemes", waitForAudioChunks: true, gain: 1 },
          undefined,
          // The library also reports "ended" when playback catches up with synthesis mid-utterance
          // (buffer underrun between sentences); only honour it once all audio has been received.
          // It can also fire before the queued audio has played out, so never end before that.
          () => { if (endRequested && performance.now() >= firstAt + queuedS * 1000 - 150) finish(utt); },
        );
        streaming = true;
        sr = sampleRate;
      }
      utt += 1;
      started = false;
      ended = false;
      endRequested = false;
      queuedS = 0;
      if (endTimer) clearTimeout(endTimer);
      endTimer = null;
    },
    push(pcm) {
      if (ended) return;
      queuedS += pcm.byteLength / 2 / sr; // measure first: streamAudio transfers (detaches) the buffer
      head.streamAudio({ audio: pcm });
      if (!started) {
        started = true;
        firstAt = performance.now();
        sp.onStarted?.();
      }
    },
    end() {
      endRequested = true;
      head.streamNotifyEnd();
      const u = utt;
      if (!started) return finish(u);
      // fallback in case the library's end callback doesn't fire for this utterance
      const remaining = firstAt + queuedS * 1000 - performance.now();
      endTimer = setTimeout(() => finish(u), Math.max(0, remaining) + 400);
    },
    interrupt() {
      head.streamInterrupt();
      ended = true;
      if (endTimer) clearTimeout(endTimer);
      endTimer = null;
    },
    setState(s) {
      if (nod) { clearInterval(nod); nod = null; }
      if (s === "listening") {
        head.setMood("neutral");
        head.makeEyeContact(60_000);
        // attentive listening: brief eye contact refreshes and the occasional glance down
        nod = setInterval(() => { head.lookAt(null, null, 300); head.makeEyeContact(8000); }, 7000);
      } else if (s === "thinking") {
        head.lookAt(window.innerWidth * 0.7, window.innerHeight * 0.25, 1400); // glance up and aside while thinking
      } else if (s === "speaking") {
        head.makeEyeContact(10_000);
      } else {
        head.lookAhead(2000);
      }
    },
    level() {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = ((buf[i] ?? 128) - 128) / 128; sum += v * v; }
      return Math.min(1, Math.sqrt(sum / buf.length) * 4);
    },
    fps: () => measured,
    async unlock() {
      if (head.audioCtx.state === "suspended") await head.audioCtx.resume().catch(() => undefined);
    },
    destroy() {
      if (nod) clearInterval(nod);
      if (endTimer) clearTimeout(endTimer);
      try { head.streamStop(); head.stop(); } catch { /* already stopped */ }
      el.replaceChildren();
    },
  };
  return sp;
}

function applyLook(head: TalkingHeadLike, look: Look) {
  head.armature?.traverse((o) => {
    const name = o.material?.name ?? "";
    if (!o.isMesh || !o.material?.color) return;
    if (look.hair && name.includes("ponytail")) o.material.color.set(look.hair);
    if (look.top && name.includes("casualsuit")) o.material.color.set(look.top);
    if (look.skin && name === "Human.body") o.material.color.set(look.skin);
  });
}

// ----------------------------------------------------------------------------- plain audio fallback

export function createAudioSpeaker(): Speaker {
  const ctx = new AudioContext({ latencyHint: "interactive" });
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.connect(ctx.destination);
  const buf = new Uint8Array(analyser.fftSize);
  let sampleRate = 24000;
  let next = 0;
  let sources: AudioBufferSourceNode[] = [];
  let ending = false;
  let started = false;
  const sp: Speaker = {
    kind: "audio",
    onStarted: null,
    onEnded: null,
    start(sr) {
      sampleRate = sr;
      ending = false;
      started = false;
      next = Math.max(next, ctx.currentTime + 0.05);
    },
    push(pcm) {
      const i16 = new Int16Array(pcm);
      if (!i16.length) return;
      const ab = ctx.createBuffer(1, i16.length, sampleRate);
      const ch = ab.getChannelData(0);
      for (let i = 0; i < i16.length; i++) ch[i] = (i16[i] ?? 0) / 32768;
      const src = ctx.createBufferSource();
      src.buffer = ab;
      src.connect(analyser);
      const at = Math.max(next, ctx.currentTime + 0.02);
      src.start(at);
      next = at + ab.duration;
      if (!started) { started = true; sp.onStarted?.(); }
      sources.push(src);
      src.onended = () => {
        sources = sources.filter((s) => s !== src);
        if (ending && sources.length === 0) { ending = false; sp.onEnded?.(); }
      };
    },
    end() {
      ending = true;
      if (sources.length === 0) { ending = false; sp.onEnded?.(); }
    },
    interrupt() {
      ending = false;
      sources.forEach((s) => { s.onended = null; try { s.stop(); } catch { /* not started */ } });
      sources = [];
      next = ctx.currentTime;
    },
    setState() { /* no avatar */ },
    level() {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = ((buf[i] ?? 128) - 128) / 128; sum += v * v; }
      return Math.min(1, Math.sqrt(sum / buf.length) * 4);
    },
    fps: () => 0,
    async unlock() {
      if (ctx.state === "suspended") await ctx.resume().catch(() => undefined);
    },
    destroy() {
      sp.interrupt();
      void ctx.close();
    },
  };
  return sp;
}
