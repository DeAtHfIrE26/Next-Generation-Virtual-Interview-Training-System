// Text -> mouth-shape timeline (port of interview_core.avatar.visemes.from_text) and a
// scheduler that returns the shape for "now". Used when TTS gives no viseme marks.

export const SHAPES = ["rest", "mbp", "fv", "ee", "aa", "oo", "td", "rw"] as const;
export type Shape = (typeof SHAPES)[number];
export type Timeline = [number, Shape][];

const LETTERS: [string, Shape][] = [
  ["th", "td"], ["sh", "td"], ["ch", "td"], ["oo", "oo"], ["ee", "ee"], ["ou", "oo"],
  ["a", "aa"], ["e", "ee"], ["i", "ee"], ["y", "ee"], ["o", "oo"], ["u", "oo"], ["w", "rw"], ["r", "rw"],
  ["m", "mbp"], ["b", "mbp"], ["p", "mbp"], ["f", "fv"], ["v", "fv"],
];

export function shapesForWord(word: string): Shape[] {
  const out: Shape[] = [];
  let i = 0;
  while (i < word.length) {
    const hit = LETTERS.find(([p]) => word.startsWith(p, i));
    if (hit) { out.push(hit[1]); i += hit[0].length; } else { out.push("td"); i += 1; }
  }
  return out;
}

export function fromText(text: string, durationMs: number): Timeline {
  const shapes: Shape[] = [];
  for (const w of text.toLowerCase().match(/[a-z']+/g) ?? []) shapes.push(...shapesForWord(w), "rest");
  if (!shapes.length) return [[0, "rest"]];
  const step = durationMs / shapes.length;
  return shapes.map((s, k) => [Math.round(k * step), s]);
}

/** Timeline for one word starting at tMs and lasting durMs (browser boundary events). */
export function forWord(word: string, tMs: number, durMs: number): Timeline {
  const shapes = shapesForWord(word.toLowerCase().replace(/[^a-z']/g, ""));
  if (!shapes.length) return [];
  const step = durMs / shapes.length;
  return shapes.map((s, k) => [Math.round(tMs + k * step), s]);
}

export function shapeAt(tl: Timeline, tMs: number): Shape {
  let cur: Shape = "rest";
  for (const [t, s] of tl) { if (t <= tMs) cur = s; else break; }
  return cur;
}
