// Landmark geometry shared with the Python core (interview_core.lipsync.mouth and
// interview_core.gaze.estimator). Parity is enforced by lib/visionMath.test.ts against
// fixtures generated from the Python implementation.

export type Pt = { x: number; y: number };

const INNER_LIP_PAIRS: [number, number][] = [[13, 14], [82, 87], [312, 317]];
const EYE_OUTER: [number, number] = [33, 263];
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const at = (lm: Pt[], i: number): Pt => lm[i] ?? { x: 0, y: 0 };

/** E4: mean inner-lip gap over inter-ocular distance. */
export function mouthAperture(lm: Pt[]): number {
  const gap = INNER_LIP_PAIRS.reduce((s, [u, l]) => s + dist(at(lm, u), at(lm, l)), 0) / INNER_LIP_PAIRS.length;
  return gap / (dist(at(lm, EYE_OUTER[0]), at(lm, EYE_OUTER[1])) + 1e-9);
}

function mean(lm: Pt[], from: number, to: number): Pt {
  let x = 0, y = 0;
  for (let i = from; i < to; i++) { x += at(lm, i).x; y += at(lm, i).y; }
  const n = to - from;
  return { x: x / n, y: y / n };
}

export interface GazeFeatures { horizontal: number; vertical: number; yaw: number; pitch: number }

/** E5: same features as interview_core.gaze.extract_features (normalised coordinates). */
export function gazeFeatures(lm: Pt[]): GazeFeatures {
  const li = mean(lm, 468, 473), ri = mean(lm, 473, 478);
  const lh = (li.x - at(lm, 33).x) / (at(lm, 133).x - at(lm, 33).x + 1e-6);
  const rh = (ri.x - at(lm, 263).x) / (at(lm, 362).x - at(lm, 263).x + 1e-6);
  const lv = (li.y - at(lm, 159).y) / (at(lm, 145).y - at(lm, 159).y + 1e-9);
  const rv = (ri.y - at(lm, 386).y) / (at(lm, 374).y - at(lm, 386).y + 1e-9);
  const mid = { x: (at(lm, 33).x + at(lm, 263).x) / 2, y: (at(lm, 33).y + at(lm, 263).y) / 2 };
  const iod = dist(at(lm, 33), at(lm, 263)) + 1e-9;
  return { horizontal: (lh + rh) / 2, vertical: (lv + rv) / 2, yaw: (at(lm, 1).x - mid.x) / iod, pitch: (at(lm, 1).y - mid.y) / iod };
}

export function onScreen(f: GazeFeatures): boolean {
  return Math.abs(f.horizontal - 0.5) <= 0.2 && Math.abs(f.vertical - 0.5) <= 0.3 && Math.abs(f.yaw) <= 0.25;
}
