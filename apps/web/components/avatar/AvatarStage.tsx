"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/components/ui";
import {
  createAudioSpeaker,
  createTalkingHeadSpeaker,
  detectQuality,
  webglAvailable,
  type AvatarState,
  type Look,
  type Speaker,
} from "@/lib/avatar/speaker";

/**
 * The interviewer stage: a real-time 3D talking head (TalkingHead + three.js) with audio-driven
 * lip-sync, or an audio-reactive orb on devices without WebGL. Calls onSpeaker once ready.
 */
export function AvatarStage({
  look,
  state,
  onSpeaker,
  className,
  name,
  compact,
  deferUntilIdle,
}: {
  look: Look;
  state: AvatarState;
  onSpeaker?: (s: Speaker) => void;
  className?: string;
  name?: string;
  compact?: boolean;
  /** Start loading the 3D model only once the page is idle (marketing pages: keeps load fast). */
  deferUntilIdle?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const speakerRef = useRef<Speaker | null>(null);
  const [progress, setProgress] = useState(0);
  const [mode, setMode] = useState<"loading" | "3d" | "orb">("loading");
  const [level, setLevel] = useState(0);
  const lookKey = JSON.stringify(look);

  useEffect(() => {
    let cancelled = false;
    if (!host.current) return;
    // Each instance renders into its own child node: in dev, React mounts effects twice and the
    // first (cancelled) instance must not tear down the second one's canvas when it finishes loading.
    const el = document.createElement("div");
    el.style.cssText = "position:absolute;inset:0";
    host.current.appendChild(el);
    (async () => {
      if (deferUntilIdle) {
        await new Promise<void>((r) => {
          const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number };
          if (w.requestIdleCallback) w.requestIdleCallback(() => r(), { timeout: 3000 });
          else setTimeout(r, 1500);
        });
        if (cancelled) return;
      }
      let sp: Speaker;
      if (webglAvailable()) {
        try {
          sp = await createTalkingHeadSpeaker(el, JSON.parse(lookKey) as Look, detectQuality(), (p) => !cancelled && setProgress(p));
          if (cancelled) return sp.destroy();
          setMode("3d");
        } catch (e) {
          console.warn("3D avatar unavailable, falling back to audio orb", e);
          sp = createAudioSpeaker();
          if (!cancelled) setMode("orb");
        }
      } else {
        sp = createAudioSpeaker();
        setMode("orb");
      }
      if (cancelled) return sp.destroy();
      speakerRef.current = sp;
      onSpeaker?.(sp);
    })();
    return () => {
      cancelled = true;
      speakerRef.current?.destroy();
      speakerRef.current = null;
      setTimeout(() => el.remove(), 0);
    };
    // the speaker is created once per look; onSpeaker is a stable callback from the room
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lookKey]);

  useEffect(() => {
    speakerRef.current?.setState(state);
  }, [state, mode]);

  // speaking ring follows the actual output level
  useEffect(() => {
    if (state !== "speaking") return;
    let raf = 0;
    const tick = () => { setLevel(speakerRef.current?.level() ?? 0); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); setLevel(0); };
  }, [state]);

  return (
    <div className={cn("relative isolate overflow-hidden rounded-[24px] border border-line bg-surface", className)}>
      <div className="pointer-events-none absolute inset-0 -z-10" style={{ background: "var(--stage-glow)" }} />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-1/3 bg-gradient-to-t from-canvas/60 to-transparent" />
      <div ref={host} className={cn("absolute inset-0 transition-opacity duration-[600ms]", mode === "3d" ? "opacity-100" : "opacity-0")} aria-hidden />

      {mode === "loading" && (
        <div className="absolute inset-0 grid place-items-center">
          <div className="flex flex-col items-center gap-3 text-[13px] text-fg-muted">
            <div className="relative size-14">
              <svg viewBox="0 0 36 36" className="size-14 -rotate-90">
                <circle cx="18" cy="18" r="16" fill="none" stroke="var(--surface-3)" strokeWidth="2.5" />
                <circle cx="18" cy="18" r="16" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeDasharray={`${(progress / 100) * 100.5} 100.5`} className="transition-[stroke-dasharray] duration-200" />
              </svg>
              <span className="absolute inset-0 grid place-items-center font-mono text-[11px] tabular">{progress}%</span>
            </div>
            {!compact && <span>Preparing your interviewer…</span>}
          </div>
        </div>
      )}

      {mode === "orb" && (
        <div className="absolute inset-0 grid place-items-center" aria-hidden>
          <div
            className="size-40 rounded-full bg-[radial-gradient(circle_at_35%_30%,var(--accent),color-mix(in_srgb,var(--live)_60%,transparent)_60%,transparent_75%)] blur-[1px] transition-transform duration-75"
            style={{ transform: `scale(${1 + level * 0.25})` }}
          />
        </div>
      )}

      {name && (
        <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full border border-line bg-canvas/60 px-3 py-1.5 text-[12px] font-medium backdrop-blur-md">
          <span
            className={cn("size-2 rounded-full transition-colors", state === "speaking" ? "bg-accent" : state === "listening" ? "bg-live" : state === "thinking" ? "bg-warn" : "bg-fg-subtle")}
            style={state === "speaking" ? { boxShadow: `0 0 0 ${2 + level * 6}px color-mix(in srgb, var(--accent) 35%, transparent)` } : undefined}
          />
          {name}
        </div>
      )}
    </div>
  );
}
