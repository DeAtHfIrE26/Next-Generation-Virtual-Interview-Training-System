import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/parity.json";
import { gazeFeatures, mouthAperture, onScreen, type Pt } from "./visionMath";
import { fromText, shapeAt } from "./visemes";

const toPts = (lm: number[][]): Pt[] => lm.map(([x, y]) => ({ x: x!, y: y! }));

describe("parity with the Python core", () => {
  it("mouth aperture and gaze features match interview_core", () => {
    for (const c of fixture.landmarks) {
      const lm = toPts(c.landmarks);
      expect(mouthAperture(lm)).toBeCloseTo(c.mouth, 9);
      const g = gazeFeatures(lm);
      expect(g.horizontal).toBeCloseTo(c.gaze.horizontal, 6);
      expect(g.vertical).toBeCloseTo(c.gaze.vertical, 6);
      expect(g.yaw).toBeCloseTo(c.gaze.yaw, 9);
      expect(onScreen(g)).toBe(c.on_screen);
    }
  });

  it("text visemes match interview_core.avatar.visemes.from_text", () => {
    for (const v of fixture.visemes) expect(fromText(v.text, v.duration)).toEqual(v.timeline);
  });

  it("shapeAt returns the latest shape at or before t", () => {
    const tl = fromText("Hello", 1000);
    expect(shapeAt(tl, 0)).toBe(tl[0]![1]);
    expect(shapeAt(tl, -5)).toBe("rest");
  });
});
