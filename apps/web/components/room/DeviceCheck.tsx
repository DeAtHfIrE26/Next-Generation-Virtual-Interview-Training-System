"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Headphones, Keyboard, Mic, MicOff, Play, RefreshCw } from "lucide-react";
import { Alert, Button, Field, LevelMeter, Select, Switch } from "@/components/ui";
import { listDevices, MEDIA_ERROR_HELP, MediaAccessError, openMedia, stopStream, type Devices, type MediaErrorKind } from "@/lib/voice/mic";

export interface DeviceChoice {
  stream: MediaStream | null; // null = type answers only
  micId?: string;
  cameraId?: string;
  speakerId?: string;
}

/**
 * Pre-join check: camera preview, microphone picker with a live level meter, speaker test, and
 * clear recovery steps for every permission or device error. Typing is always available.
 */
export function DeviceCheck({ onJoin, joining, ready }: { onJoin: (c: DeviceChoice) => void; joining: boolean; ready: boolean }) {
  const [devices, setDevices] = useState<Devices>({ mics: [], cameras: [], speakers: [] });
  const [micId, setMicId] = useState<string>("");
  const [cameraId, setCameraId] = useState<string>("");
  const [speakerId, setSpeakerId] = useState<string>("");
  const [useCamera, setUseCamera] = useState(true);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<MediaErrorKind | null>(null);
  const [requesting, setRequesting] = useState(false);
  const [level, setLevel] = useState(0);
  const [heard, setHeard] = useState(false);
  const [speakerPlayed, setSpeakerPlayed] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const handedOver = useRef(false); // the stream now belongs to the room: don't stop it on unmount

  async function request(nextMic = micId, nextCam = cameraId, cam = useCamera) {
    setRequesting(true);
    setError(null);
    try {
      const s = await openMedia({ micId: nextMic || undefined, cameraId: nextCam || undefined, video: cam });
      stopStream(streamRef.current);
      streamRef.current = s;
      setStream(s);
      const d = await listDevices();
      setDevices(d);
      const track = s.getAudioTracks()[0];
      if (track && !nextMic) setMicId(track.getSettings().deviceId ?? "");
      const vt = s.getVideoTracks()[0];
      if (vt && !nextCam) setCameraId(vt.getSettings().deviceId ?? "");
    } catch (e) {
      setError(e instanceof MediaAccessError ? e.kind : "unknown");
    } finally {
      setRequesting(false);
    }
  }

  // Ask for devices as soon as the check opens.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- starts an async permission request
    void request();
    const onChange = () => void listDevices().then(setDevices);
    navigator.mediaDevices?.addEventListener?.("devicechange", onChange);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", onChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => { if (streamRef.current && !handedOver.current) stopStream(streamRef.current); }, []);

  // camera preview
  useEffect(() => {
    if (video.current) {
      video.current.srcObject = stream && stream.getVideoTracks().length ? stream : null;
      if (stream) void video.current.play().catch(() => undefined);
    }
  }, [stream]);

  // live mic level
  useEffect(() => {
    if (!stream || !stream.getAudioTracks().length) return;
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const an = ctx.createAnalyser();
    an.fftSize = 1024;
    src.connect(an);
    const buf = new Float32Array(an.fftSize);
    let raf = 0;
    const tick = () => {
      an.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += (buf[i] ?? 0) ** 2;
      const db = 20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-8);
      const l = Math.max(0, Math.min(1, (db + 60) / 50));
      setLevel(l);
      if (l > 0.35) setHeard(true);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); void ctx.close(); };
  }, [stream]);

  async function testSpeaker() {
    const ctx = new AudioContext();
    const dest = ctx.createMediaStreamDestination();
    const el = new Audio();
    const sink = el as HTMLAudioElement & { setSinkId?: (id: string) => Promise<void> };
    if (speakerId && sink.setSinkId) await sink.setSinkId(speakerId).catch(() => undefined);
    // a short two-note chime
    const now = ctx.currentTime;
    [660, 880].forEach((f, i) => {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.frequency.value = f;
      g.gain.setValueAtTime(0, now + i * 0.18);
      g.gain.linearRampToValueAtTime(0.2, now + i * 0.18 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, now + i * 0.18 + 0.35);
      o.connect(g).connect(dest);
      o.start(now + i * 0.18);
      o.stop(now + i * 0.18 + 0.4);
    });
    el.srcObject = dest.stream;
    await el.play().catch(() => undefined);
    setSpeakerPlayed(true);
    setTimeout(() => { el.pause(); void ctx.close(); }, 900);
  }

  const hasMic = !!stream?.getAudioTracks().length;
  const hasCam = !!stream?.getVideoTracks().length;
  const help = error ? MEDIA_ERROR_HELP[error] : null;

  return (
    <div className="flex flex-col gap-5">
      <div className="relative aspect-video overflow-hidden rounded-[16px] border border-line bg-surface-2 lg:aspect-auto lg:h-[200px]">
        <video ref={video} muted playsInline className="size-full -scale-x-100 object-cover" aria-label="Your camera preview" />
        {!hasCam && (
          <div className="absolute inset-0 grid place-items-center text-center text-[13px] text-fg-muted">
            <div className="flex flex-col items-center gap-2">
              <CameraOff className="size-6" />
              {useCamera ? "Camera unavailable. You can still interview with audio only." : "Camera off"}
            </div>
          </div>
        )}
        <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full bg-canvas/70 px-3 py-1.5 text-[12px] backdrop-blur">
          {hasMic ? <Mic className="size-3.5 text-live" /> : <MicOff className="size-3.5 text-danger" />}
          <LevelMeter level={level} bars={10} className="h-3" />
        </div>
      </div>

      {help && (
        <Alert tone="warn" title={help.title} action={<Button size="sm" variant="secondary" onClick={() => void request()} loading={requesting}><RefreshCw className="size-3.5" /> Try again</Button>}>
          {help.body}
        </Alert>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Microphone" htmlFor="mic">
          <Select id="mic" value={micId} onChange={(e) => { setMicId(e.target.value); void request(e.target.value); }} disabled={!devices.mics.length}>
            {!devices.mics.length && <option value="">No microphone found</option>}
            {devices.mics.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Microphone ${i + 1}`}</option>)}
          </Select>
        </Field>
        <Field label="Camera" htmlFor="cam" hint={<span>Only used on your device for eye contact and integrity checks.</span>}>
          <div className="flex items-center gap-3">
            <Select id="cam" value={cameraId} onChange={(e) => { setCameraId(e.target.value); void request(micId, e.target.value, true); }} disabled={!useCamera || !devices.cameras.length} className="flex-1">
              {!devices.cameras.length && <option value="">No camera found</option>}
              {devices.cameras.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
            </Select>
            <Switch checked={useCamera} label="Use camera" onChange={(v) => { setUseCamera(v); void request(micId, cameraId, v); }} />
          </div>
        </Field>
        {devices.speakers.length > 0 && (
          <Field label="Speaker" htmlFor="spk">
            <Select id="spk" value={speakerId} onChange={(e) => setSpeakerId(e.target.value)}>
              {devices.speakers.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Speaker ${i + 1}`}</option>)}
            </Select>
          </Field>
        )}
        <div className="flex items-end">
          <Button variant="secondary" onClick={() => void testSpeaker()} className="w-full"><Play className="size-4" /> {speakerPlayed ? "Play test sound again" : "Test speakers"}</Button>
        </div>
      </div>

      <ul className="flex flex-col gap-2 text-[13px]">
        <Check ok={hasMic} label={hasMic ? (heard ? "We can hear you" : "Say something to test your microphone") : "Microphone not connected"} icon={<Mic className="size-3.5" />} />
        <Check ok={hasCam || !useCamera} label={hasCam ? "Camera ready" : useCamera ? "Camera unavailable (optional)" : "Camera off (optional)"} icon={<Camera className="size-3.5" />} optional />
        <Check ok={speakerPlayed} label={speakerPlayed ? "Speakers tested" : "Test your speakers, or use headphones to avoid echo"} icon={<Headphones className="size-3.5" />} optional />
      </ul>

      <div className="flex flex-col gap-2 sm:flex-row">
        <Button
          size="lg"
          className="flex-1"
          disabled={!ready || joining || (!hasMic && !error)}
          loading={joining}
          data-testid="join"
          onClick={() => { handedOver.current = true; onJoin({ stream, micId, cameraId, speakerId }); }}
        >
          {ready ? "Join interview" : "Preparing interviewer…"}
        </Button>
        <Button
          size="lg"
          variant="secondary"
          disabled={!ready || joining}
          onClick={() => { stopStream(streamRef.current); handedOver.current = true; onJoin({ stream: null }); }}
        >
          <Keyboard className="size-4" /> Type answers instead
        </Button>
      </div>
    </div>
  );
}

function Check({ ok, label, icon, optional }: { ok: boolean; label: string; icon: React.ReactNode; optional?: boolean }) {
  return (
    <li className="flex items-center gap-2">
      <span className={`grid size-5 place-items-center rounded-full ${ok ? "bg-live/15 text-live" : optional ? "bg-surface-3 text-fg-subtle" : "bg-warn/15 text-warn"}`}>{icon}</span>
      <span className={ok ? "text-fg" : "text-fg-muted"}>{label}</span>
    </li>
  );
}
