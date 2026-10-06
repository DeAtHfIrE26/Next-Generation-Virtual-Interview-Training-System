"use client";

import { useEffect, useState } from "react";
import { percentile, type RoomController, type RoomState } from "@/lib/room/controller";

/** ?debug=1 diagnostics: connection, providers, per-stage latency, audio and avatar health. */
export function DiagPanel({ ctl, state }: { ctl: RoomController; state: RoomState }) {
  const [, force] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => force((n) => n + 1), 500);
    return () => clearInterval(iv);
  }, []);
  const d = ctl.diag;
  const rows: [string, string][] = [
    ["connection", `${state.conn}${state.connDetail ? ` (${state.connDetail})` : ""}`],
    ["phase", state.phase],
    ["rtt", `${d.rttMs.toFixed(0)} ms`],
    ["reconnects", String(d.reconnects)],
    ["llm", `${state.providers?.llm ?? "-"} ${state.providers?.llm_model ?? ""}${state.providers?.llm_fallback ? ` → ${state.providers.llm_fallback}` : ""}`],
    ["stt / tts", `${state.providers?.stt ?? "-"} / ${state.providers?.tts ?? "-"}`],
    ["vad", `${state.vadKind} p=${d.vadProb.toFixed(2)}${state.userSpeaking ? " speaking" : ""}`],
    ["mic", `level ${d.micLevel.toFixed(2)} · ${d.sampleRate} Hz · ctx ${d.audioCtx} · frames ${d.framesSent}`],
    ["avatar", `${d.speakerKind} · ${d.fps.toFixed(0)} fps`],
    ["barge-ins", String(d.bargeIns)],
    ["emergency questions", String(state.emergencyCount)],
  ];
  const stages = Object.entries(d.stages);
  return (
    <aside className="fixed right-3 top-16 z-50 w-[320px] rounded-[14px] border border-line bg-canvas/90 p-3 font-mono text-[11px] leading-4 shadow-[var(--shadow-float)] backdrop-blur" data-testid="diag">
      <p className="mb-2 font-sans text-[12px] font-semibold">Diagnostics</p>
      <table className="w-full">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}><td className="pr-2 align-top text-fg-subtle">{k}</td><td className="break-all">{v}</td></tr>
          ))}
        </tbody>
      </table>
      {stages.length > 0 && (
        <table className="mt-2 w-full">
          <thead><tr className="text-fg-subtle"><td>stage</td><td>n</td><td>p50</td><td>p95</td><td>last</td></tr></thead>
          <tbody>
            {stages.map(([k, xs]) => (
              <tr key={k}><td>{k}</td><td>{xs.length}</td><td>{percentile(xs, 50).toFixed(0)}</td><td>{percentile(xs, 95).toFixed(0)}</td><td>{(xs.at(-1) ?? 0).toFixed(0)}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </aside>
  );
}
