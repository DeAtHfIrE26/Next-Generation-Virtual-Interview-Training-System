"use client";

// Interviewer avatar: an original 2D vector rig driven by viseme timelines (TTS viseme marks,
// browser word-boundary events, or a text estimate). Not a patent element; it needs no GPU,
// so the first animated frame does not wait on any model. A neural talking-head stream can
// replace it behind FEATURE_NEURAL_AVATAR; any failure falls back to this rig, and
// prefers-reduced-motion falls back to captions with a static face.
import { useEffect, useRef, useState } from "react";
import { shapeAt, type Shape, type Timeline } from "@/lib/visemes";

const MOUTHS: Record<Shape, string> = {
  rest: "M84 132 Q100 136 116 132",
  mbp: "M84 133 Q100 133 116 133",
  fv: "M84 131 Q100 140 116 131 Q100 135 84 131",
  ee: "M80 130 Q100 142 120 130 Q100 136 80 130",
  aa: "M86 128 Q100 152 114 128 Q100 132 86 128",
  oo: "M92 128 Q100 146 108 128 Q100 124 92 128",
  td: "M84 130 Q100 142 116 130 Q100 134 84 130",
  rw: "M90 129 Q100 141 110 129 Q100 127 90 129",
};

export interface AvatarProps {
  timeline: Timeline;
  startedAt: number | null; // performance.now() when speech started, null when silent
  onFirstFrame?: (t: number) => void;
  label?: string;
}

export function Avatar({ timeline, startedAt, onFirstFrame, label = "Interviewer" }: AvatarProps) {
  const [shape, setShape] = useState<Shape>("rest");
  const [blink, setBlink] = useState(false);
  const reported = useRef<number | null>(null);
  const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (startedAt === null || reduced) return;
    let raf = 0;
    const tick = () => {
      const s = shapeAt(timeline, performance.now() - startedAt);
      setShape((prev) => (prev === s ? prev : s));
      if (s !== "rest" && reported.current !== startedAt) {
        reported.current = startedAt;
        onFirstFrame?.(performance.now());
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [timeline, startedAt, onFirstFrame, reduced]);

  useEffect(() => {
    if (reduced) return;
    const id = setInterval(() => { setBlink(true); setTimeout(() => setBlink(false), 140); }, 3800 + Math.random() * 1800);
    return () => clearInterval(id);
  }, [reduced]);

  const mouth: Shape = startedAt === null || reduced ? "rest" : shape;
  return (
    <svg viewBox="0 0 200 200" role="img" aria-label={`${label}${startedAt !== null ? " (speaking)" : ""}`}
      style={{ width: "100%", maxWidth: 280, display: "block", margin: "0 auto" }}>
      <circle cx="100" cy="100" r="96" fill="var(--surface-2)" />
      <path d="M44 112 Q40 40 100 36 Q160 40 156 112 Q150 70 100 66 Q50 70 44 112Z" fill="#3b3a36" />
      <ellipse cx="100" cy="108" rx="50" ry="58" fill="#d9a77c" />
      <path d="M50 96 Q54 58 100 56 Q146 58 150 96 Q140 72 100 72 Q60 72 50 96Z" fill="#3b3a36" />
      <path d="M70 92 Q78 88 86 92" stroke="#3b3a36" strokeWidth="3" fill="none" strokeLinecap="round" />
      <path d="M114 92 Q122 88 130 92" stroke="#3b3a36" strokeWidth="3" fill="none" strokeLinecap="round" />
      {blink ? (
        <g stroke="#2b2a27" strokeWidth="3" strokeLinecap="round"><line x1="72" y1="104" x2="84" y2="104" /><line x1="116" y1="104" x2="128" y2="104" /></g>
      ) : (
        <g fill="#2b2a27"><circle cx="78" cy="104" r="4.5" /><circle cx="122" cy="104" r="4.5" /></g>
      )}
      <path d="M100 108 Q96 118 100 120" stroke="#b07f58" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d={MOUTHS[mouth]} fill={mouth === "rest" || mouth === "mbp" ? "none" : "#7a2f2f"} stroke="#7a2f2f"
        strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
      <path d="M58 166 Q100 150 142 166 L150 200 L50 200Z" fill="var(--accent)" />
    </svg>
  );
}

/** Optional neural avatar: plays a stream URL; reports failure so the caller can fall back. */
export function NeuralAvatar({ src, onError, onEnded, onPlaying }: { src: string; onError: () => void; onEnded: () => void;
  onPlaying: () => void }) {
  return <video src={src} autoPlay playsInline onError={onError} onEnded={onEnded} onPlaying={onPlaying}
    aria-label="Interviewer (video)" style={{ width: "100%", borderRadius: 10 }} />;
}
