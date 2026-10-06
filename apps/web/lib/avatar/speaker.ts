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
  /** Mouth-shape updates driven by the playing audio since load (0 for the audio-only fallback). */
  visemeUpdates(): number;
  debug?: () => Record<string, unknown>;
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
  opt: { update: ((dt: number) => void) | null; modelFPS: number; modelPixelRatio: number };
  animFrameDur: number;
  workletLoaded?: boolean;
  armature?: { traverse(fn: (o: { isMesh?: boolean; material?: { name?: string; color?: { set(c: string): void } } }) => void): void };
  initAudioGraph?(sampleRate?: number | null): void;
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

/** Sample rate of the default TTS (Kokoro). Other providers trigger an automatic re-wire. */
const TTS_SAMPLE_RATE = 24000;
const PLAYBACK_WORKLET = "/vendor/talkinghead/playback-worklet.js";

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
  // TalkingHead rebuilds its AudioContext (and every node) when a stream's sample rate differs
  // from the context's, so the graph is created at the TTS rate up front and HeadAudio is re-wired
  // whenever the context changes; otherwise it would listen to a dead graph and the mouth would
  // not move (found by the viseme counter in the E2E diagnostics).
  const { HeadAudio } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ haUrl);
  let visemeUpdates = 0;
  const dbg = { calls: 0, max: 0, keys: new Set<string>() };
  let ha: { update(dt: number): void; onvalue: ((k: string, v: number) => void) | null; disconnect(): void } | null = null;
  let wiredCtx: AudioContext | null = null;
  let wiring: Promise<void> | null = null;
  const wire = async () => {
    const ctx = head.audioCtx;
    if (wiredCtx === ctx) return;
    wiredCtx = ctx;
    await ctx.audioWorklet.addModule("/vendor/headaudio/headworklet.min.mjs");
    const node = new HeadAudio(ctx, { processorOptions: {}, parameterData: { vadGateActiveDb: -45, vadGateInactiveDb: -60, speakerMeanHz: 210 } });
    await node.loadModel("/vendor/headaudio/model-en-mixed.bin"); // browser-cached after the first load
    if (head.audioCtx !== ctx) return; // replaced again meanwhile
    head.audioStreamGainNode.connect(node);
    node.onvalue = (key: string, value: number) => {
      const t = head.mtAvatar[key];
      if (t) Object.assign(t, { newvalue: value, needsUpdate: true });
      dbg.calls += 1;
      dbg.max = Math.max(dbg.max, value);
      if (t) dbg.keys.add(key);
    };
    // Count the visemes the worklet detects in the played audio (not rendered frames, which a busy
    // device draws at 1-2 fps): the lip-sync signal itself. Runs after HeadAudio's own handler.
    node.port.addEventListener("message", (e: MessageEvent) => {
      if (e.data?.event === "viseme" && node.visemeActive !== -1) visemeUpdates += 1;
    });
    try { ha?.disconnect(); } catch { /* old context closed */ }
    ha = node;
  };
  const ensureWired = () => {
    if (wiredCtx !== head.audioCtx) wiring = wire().catch((e) => console.warn("lip-sync wiring failed", e));
    return wiring;
  };
  head.initAudioGraph?.(TTS_SAMPLE_RATE);
  // TalkingHead loads its playback worklet on the first streamStart with a 5 s timeout, which a
  // busy device can miss (the interviewer would then be silent). Load it here, without a timeout,
  // while the avatar is loading anyway.
  const loadPlayback = async () => {
    await head.audioCtx.audioWorklet.addModule(PLAYBACK_WORKLET);
    head.workletLoaded = true;
  };
  await loadPlayback();
  await ensureWired();
  let frames = 0;
  let last = performance.now();
  let measured = quality.fps;
  // Adaptive quality: if the device can't hold half the target frame rate for 3 s, render fewer,
  // smaller frames (up to twice). The avatar shares the main thread with audio capture, barge-in
  // detection and the UI; those must not starve behind the renderer.
  let slowSeconds = 0;
  let degraded = 0;
  let pixelRatio = quality.dpr;
  head.opt.update = (dt: number) => {
    ha?.update(dt);
    frames += 1;
    const now = performance.now();
    if (now - last >= 1000) {
      measured = (frames * 1000) / (now - last);
      frames = 0;
      last = now;
      slowSeconds = measured < head.opt.modelFPS * 0.5 ? slowSeconds + 1 : 0;
      if (slowSeconds >= 3 && degraded < 2) {
        degraded += 1;
        slowSeconds = 0;
        const dpr = window.devicePixelRatio || 1;
        pixelRatio = Math.max(0.25 / dpr, pixelRatio * 0.6); // a multiplier of devicePixelRatio, as in TalkingHead
        head.opt.modelPixelRatio = pixelRatio; // kept on resize
        head.renderer?.setPixelRatio(pixelRatio * dpr);
        head.opt.modelFPS = Math.max(5, Math.round(head.opt.modelFPS * 0.6));
        head.animFrameDur = 1000 / head.opt.modelFPS;
      }
    }
  };

  const buf = new Uint8Array(256);
  let streaming = false;
  let sr = 0;
  // Per-utterance bookkeeping. TalkingHead's stream callbacks fire once per stream session, not per
  // utterance, so start/end are tracked here from the audio actually queued.
  let utt = 0;
  let started = false;
  let ended = true;
  let endRequested = false; // tts.end received: no more audio is coming for this utterance
  let queuedS = 0;
  // When synthesis runs slower than real time, playback stalls between chunks, so the audio ends
  // later than first chunk + queuedS. playEnd models the play-out: each chunk starts when it arrives or
  // when the previous one finishes, whichever is later.
  let playEnd = 0;
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
        // The library also reports "ended" when playback catches up with synthesis mid-utterance
        // (buffer underrun between sentences); only honour it once all audio has been received.
        // It can also fire before the queued audio has played out, so never end before that.
        const onStreamEnd = () => { if (endRequested && performance.now() >= playEnd - 150) finish(utt); };
        const stream = head.streamStart({ sampleRate, lipsyncType: "visemes", waitForAudioChunks: true, gain: 1 }, undefined, onStreamEnd);
        void Promise.resolve(stream)
          .catch(async () => {
            // worklet load timed out inside the library: load it without a timeout and retry once
            await loadPlayback();
            await head.streamStart({ sampleRate, lipsyncType: "visemes", waitForAudioChunks: true, gain: 1 }, undefined, onStreamEnd);
          })
          .then(() => ensureWired()) // the context may have been rebuilt
          .catch((e) => console.warn("interviewer audio stream failed", e));
        streaming = true;
        sr = sampleRate;
      }
      utt += 1;
      started = false;
      ended = false;
      endRequested = false;
      queuedS = 0;
      playEnd = 0;
      if (endTimer) clearTimeout(endTimer);
      endTimer = null;
    },
    push(pcm) {
      if (ended) return;
      const dur = pcm.byteLength / 2 / sr; // measure first: streamAudio transfers (detaches) the buffer
      queuedS += dur;
      playEnd = Math.max(playEnd, performance.now()) + dur * 1000;
      head.streamAudio({ audio: pcm });
      if (!started) {
        started = true;
        sp.onStarted?.();
      }
    },
    end() {
      endRequested = true;
      head.streamNotifyEnd();
      const u = utt;
      if (!started) return finish(u);
      // fallback in case the library's end callback doesn't fire for this utterance
      const remaining = playEnd - performance.now();
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
      head.audioAnalyzerNode.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = ((buf[i] ?? 128) - 128) / 128; sum += v * v; }
      return Math.min(1, Math.sqrt(sum / buf.length) * 4);
    },
    fps: () => measured,
    visemeUpdates: () => visemeUpdates,
    debug: () => {
      const buf2 = new Uint8Array(head.audioAnalyzerNode.fftSize);
      head.audioAnalyzerNode.getByteTimeDomainData(buf2);
      let peak = 0;
      for (const v of buf2) peak = Math.max(peak, Math.abs(v - 128));
      return { ctx: head.audioCtx.state, peak, haCalls: dbg.calls, haMax: dbg.max, keys: [...dbg.keys].slice(0, 5), streaming, degraded, modelFPS: head.opt.modelFPS };
    },
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
    visemeUpdates: () => 0,
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
