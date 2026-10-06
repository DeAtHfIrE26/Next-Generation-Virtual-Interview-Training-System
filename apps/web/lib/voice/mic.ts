// Microphone (and optional camera) capture with explicit error states, device selection,
// browser echo cancellation / noise suppression / AGC, and 16 kHz PCM16 frames for the server.

export type MediaErrorKind = "denied" | "no_device" | "in_use" | "insecure" | "unsupported" | "unknown";

export class MediaAccessError extends Error {
  constructor(public kind: MediaErrorKind, message: string) {
    super(message);
  }
}

export const MEDIA_ERROR_HELP: Record<MediaErrorKind, { title: string; body: string }> = {
  denied: {
    title: "Microphone access is blocked",
    body: "Allow microphone access for this site in your browser's address bar (the camera/mic icon), then try again. You can also type your answers.",
  },
  no_device: { title: "No microphone found", body: "Connect a microphone or headset, then choose it from the device list." },
  in_use: {
    title: "Your microphone is busy",
    body: "Another app (a video call, a recorder) is using it. Close that app and try again.",
  },
  insecure: { title: "This page isn't secure", body: "Microphone access needs HTTPS. Open the site over https://." },
  unsupported: { title: "This browser can't capture audio", body: "Use a recent version of Chrome, Edge, Firefox or Safari." },
  unknown: { title: "Couldn't start the microphone", body: "Try again, or pick another device. You can also type your answers." },
};

export function classifyMediaError(e: unknown): MediaAccessError {
  const name = (e as { name?: string })?.name ?? "";
  const msg = (e as { message?: string })?.message ?? String(e);
  if (name === "NotAllowedError" || name === "PermissionDeniedError") return new MediaAccessError("denied", msg);
  if (name === "NotFoundError" || name === "DevicesNotFoundError" || name === "OverconstrainedError") return new MediaAccessError("no_device", msg);
  if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") return new MediaAccessError("in_use", msg);
  if (name === "SecurityError") return new MediaAccessError("insecure", msg);
  return new MediaAccessError("unknown", msg);
}

export interface Devices {
  mics: MediaDeviceInfo[];
  cameras: MediaDeviceInfo[];
  speakers: MediaDeviceInfo[];
}

export async function listDevices(): Promise<Devices> {
  if (!navigator.mediaDevices?.enumerateDevices) return { mics: [], cameras: [], speakers: [] };
  const all = await navigator.mediaDevices.enumerateDevices();
  return {
    mics: all.filter((d) => d.kind === "audioinput"),
    cameras: all.filter((d) => d.kind === "videoinput"),
    speakers: all.filter((d) => d.kind === "audiooutput"),
  };
}

export async function micPermission(): Promise<PermissionState | "unknown"> {
  try {
    const p = await navigator.permissions.query({ name: "microphone" as PermissionName });
    return p.state;
  } catch {
    return "unknown";
  }
}

export async function openMedia(opts: { micId?: string; cameraId?: string; video: boolean }): Promise<MediaStream> {
  if (!window.isSecureContext) throw new MediaAccessError("insecure", "insecure context");
  if (!navigator.mediaDevices?.getUserMedia) throw new MediaAccessError("unsupported", "getUserMedia missing");
  const audio: MediaTrackConstraints = {
    deviceId: opts.micId ? { exact: opts.micId } : undefined,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
  };
  const video: MediaTrackConstraints | false = opts.video
    ? { deviceId: opts.cameraId ? { exact: opts.cameraId } : undefined, width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
    : false;
  try {
    return await navigator.mediaDevices.getUserMedia({ audio, video });
  } catch (e) {
    if (opts.video) {
      // A camera problem must not block the interview: retry audio-only.
      try {
        return await navigator.mediaDevices.getUserMedia({ audio, video: false });
      } catch (e2) {
        throw classifyMediaError(e2);
      }
    }
    throw classifyMediaError(e);
  }
}

/** Streams 20 ms 16 kHz PCM16 frames and an RMS level from a MediaStream's audio track. */
export class MicCapture {
  private node: AudioWorkletNode | null = null;
  private src: MediaStreamAudioSourceNode | null = null;
  private sink: GainNode | null = null;
  level = 0;

  private constructor(public ctx: AudioContext, public stream: MediaStream) {}

  static async start(stream: MediaStream, onFrame: (pcm: ArrayBuffer) => void): Promise<MicCapture> {
    const ctx = new AudioContext({ latencyHint: "interactive" });
    await ctx.audioWorklet.addModule("/worklets/pcm16k.js");
    const cap = new MicCapture(ctx, stream);
    cap.src = ctx.createMediaStreamSource(stream);
    cap.node = new AudioWorkletNode(ctx, "pcm16k");
    cap.sink = ctx.createGain();
    cap.sink.gain.value = 0; // keep the graph pulling without playing the mic back
    cap.node.port.onmessage = (e: MessageEvent<{ pcm: ArrayBuffer; rms: number }>) => {
      const db = 20 * Math.log10(e.data.rms + 1e-8);
      cap.level = Math.max(0, Math.min(1, (db + 60) / 50)); // -60 dB .. -10 dB -> 0..1
      onFrame(e.data.pcm);
    };
    cap.src.connect(cap.node).connect(cap.sink).connect(ctx.destination);
    // resume() may never settle without an audio device; capture still runs once it does
    if (ctx.state === "suspended") void ctx.resume().catch(() => undefined);
    return cap;
  }

  setMuted(muted: boolean) {
    this.node?.port.postMessage({ muted });
    this.stream.getAudioTracks().forEach((t) => (t.enabled = !muted));
  }

  async stop() {
    this.node?.port.close();
    this.src?.disconnect();
    this.node?.disconnect();
    await this.ctx.close().catch(() => undefined);
  }
}

export function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((t) => t.stop());
}
