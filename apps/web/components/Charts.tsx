"use client";

// Single-series charts in inline SVG. One validated series colour per mode (--series-1),
// recessive hairline grid, 2px line with ringed markers, <=24px bars with a 4px rounded data
// end, hover/focus tooltip, and a table view so no value depends on hover or colour.
import { useState } from "react";

export interface Point { label: string; value: number | null; note?: string }

const W = 640;

export function ScoreTrend({ points, title }: { points: Point[]; title: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const pts = points.filter((p) => p.value !== null) as (Point & { value: number })[];
  if (pts.length === 0) return <p className="muted">No finished sessions yet.</p>;
  const H = 220, L = 40, R = 16, T = 12, B = 28;
  const x = (i: number) => L + (pts.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (pts.length - 1));
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1]!;
  const active = hover !== null ? pts[hover] : null;
  return (
    <figure className="viz" style={{ margin: 0 }}>
      <figcaption className="sr-only">{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. Latest ${Math.round(last.value * 100)}%.`}
        onPointerLeave={() => setHover(null)}
        onPointerMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          let best = 0;
          pts.forEach((_, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i; });
          setHover(best);
        }}>
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--grid)" strokeWidth={1} />
            <text x={L - 8} y={y(g) + 4} textAnchor="end" fontSize={11} fill="var(--text-3)">{g * 100}%</text>
          </g>
        ))}
        {active && <line x1={x(hover!)} x2={x(hover!)} y1={T} y2={H - B} stroke="var(--text-3)" strokeWidth={1} />}
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {pts.map((p, i) => (
          <circle key={i} cx={x(i)} cy={y(p.value)} r={i === hover || i === pts.length - 1 ? 5 : 4}
            fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} tabIndex={0}
            aria-label={`${p.label}: ${Math.round(p.value * 100)}%`} onFocus={() => setHover(i)} onBlur={() => setHover(null)} />
        ))}
        <text x={x(pts.length - 1)} y={y(last.value) - 10} textAnchor="end" fontSize={12} fill="var(--text)">
          {Math.round(last.value * 100)}%
        </text>
        <text x={L} y={H - 6} fontSize={11} fill="var(--text-3)">{pts[0]!.label}</text>
        {pts.length > 1 && <text x={W - R} y={H - 6} textAnchor="end" fontSize={11} fill="var(--text-3)">{last.label}</text>}
      </svg>
      {active && (
        <div className="tip" style={{ left: `${(x(hover!) / W) * 100}%`, top: 0, transform: "translateX(-50%)" }}>
          <strong>{Math.round(active.value * 100)}%</strong><span>{active.label}{active.note ? ` · ${active.note}` : ""}</span>
        </div>
      )}
      <details className="table-view"><summary>Show as table</summary>
        <table className="data"><thead><tr><th>Session</th><th>Overall</th></tr></thead>
          <tbody>{pts.map((p, i) => <tr key={i}><td>{p.label}</td><td>{Math.round(p.value * 100)}%</td></tr>)}</tbody></table>
      </details>
    </figure>
  );
}

export function DimensionBars({ items, max = 5, title }: { items: Point[]; max?: number; title: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const L = 150, R = 48, row = 36, bar = 20;
  const H = items.length * row + 8;
  const x = (v: number) => (v / max) * (W - L - R);
  return (
    <figure className="viz" style={{ margin: 0 }}>
      <figcaption className="sr-only">{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {[1, 2, 3, 4, 5].map((g) => (
          <line key={g} x1={L + x(g)} x2={L + x(g)} y1={0} y2={H - 8} stroke="var(--grid)" strokeWidth={1} />
        ))}
        {items.map((it, i) => {
          const yy = i * row + (row - bar) / 2;
          const w = it.value === null ? 0 : x(it.value);
          const r = Math.min(4, w / 2);
          return (
            <g key={it.label} tabIndex={0} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}
              onFocus={() => setHover(i)} onBlur={() => setHover(null)}
              aria-label={`${it.label}: ${it.value === null ? "not scored" : `${it.value.toFixed(1)} of ${max}`}`}>
              <rect x={0} y={i * row} width={W} height={row} fill="transparent" />
              <text x={L - 10} y={yy + bar / 2 + 4} textAnchor="end" fontSize={13} fill="var(--text-2)">{it.label}</text>
              {it.value !== null && w > 0 && (
                <path d={`M${L},${yy} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${bar - 2 * r} a${r},${r} 0 0 1 -${r},${r} h-${w - r} z`}
                  fill="var(--series-1)" />
              )}
              <text x={L + w + 8} y={yy + bar / 2 + 4} fontSize={13} fill="var(--text)">
                {it.value === null ? "not scored" : it.value.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover !== null && items[hover] && (
        <div className="tip" style={{ left: L, top: hover * row - 4 }}>
          <strong>{items[hover]!.value === null ? "not scored" : `${items[hover]!.value!.toFixed(2)} / ${max}`}</strong>
          <span>{items[hover]!.label}{items[hover]!.note ? ` · ${items[hover]!.note}` : ""}</span>
        </div>
      )}
      <details className="table-view"><summary>Show as table</summary>
        <table className="data"><thead><tr><th>Dimension</th><th>Average (1-5)</th></tr></thead>
          <tbody>{items.map((p) => <tr key={p.label}><td>{p.label}</td><td>{p.value === null ? "not scored" : p.value.toFixed(2)}</td></tr>)}</tbody></table>
      </details>
    </figure>
  );
}
