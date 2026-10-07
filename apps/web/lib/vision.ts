// On-device vision with MediaPipe Tasks (Apache-2.0): face landmarks (E2/E4/E5/E7) and phone
// detection (E7). Raw video never leaves the browser; only numbers and, for enrolment and
// periodic verification, small face crops are sent.
import type { FaceLandmarker, ObjectDetector } from "@mediapipe/tasks-vision";
import { gazeFeatures, mouthAperture, onScreen, type Pt } from "./visionMath";

export interface VisionSample {
  t: number; // performance.now() ms
  faces: number;
  landmarks: Pt[] | null;
  aperture: number | null;
  onScreen: boolean | null;
  phone: boolean | null; // null when not evaluated this frame
}

export class VisionMonitor {
  private raf = 0;
  private lastFace = 0;
  private lastObj = 0;
  private failures = 0;
  private constructor(private face: FaceLandmarker, private objects: ObjectDetector | null) {}

  static async create(): Promise<VisionMonitor> {
    const { FilesetResolver, FaceLandmarker, ObjectDetector } = await import("@mediapipe/tasks-vision");
    const fs = await FilesetResolver.forVisionTasks("/vision/wasm");
    const make = async (delegate: "GPU" | "CPU") => FaceLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "/vision/face_landmarker.task", delegate },
      runningMode: "VIDEO", numFaces: 2,
    });
    // The GPU delegate needs a hardware WebGL2 context. Without one (software GL, or engines that
    // fall back to WebGL 1) it throws on every frame and can wedge the page on close: use the CPU.
    const face = gpuUsable() ? await make("GPU").catch(() => make("CPU")) : await make("CPU");
    const objects = await ObjectDetector.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "/vision/efficientdet_lite0.tflite", delegate: "CPU" },
      runningMode: "VIDEO", scoreThreshold: 0.4, maxResults: 3, categoryAllowlist: ["cell phone"],
    }).catch(() => null);
    return new VisionMonitor(face, objects);
  }

  get canDetectPhones() { return this.objects !== null; }

  start(video: HTTPVideoLike, onSample: (s: VisionSample) => void, faceFps = 12, objectFps = 2) {
    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      const now = performance.now();
      if (video.readyState < 2 || now - this.lastFace < 1000 / faceFps) return;
      this.lastFace = now;
      let r: ReturnType<FaceLandmarker["detectForVideo"]>;
      let phone: boolean | null = null;
      try {
        r = this.face.detectForVideo(video as HTMLVideoElement, now);
        if (this.objects && now - this.lastObj >= 1000 / objectFps) {
          this.lastObj = now;
          phone = this.objects.detectForVideo(video as HTMLVideoElement, now).detections.length > 0;
        }
        this.failures = 0;
      } catch (e) {
        if (++this.failures >= 10) {
          console.warn("on-device vision stopped after repeated errors", e);
          this.stop();
        }
        return;
      }
      const first = r.faceLandmarks[0]?.map((p) => ({ x: p.x, y: p.y })) ?? null;
      onSample({
        t: now, faces: r.faceLandmarks.length, landmarks: first,
        aperture: first ? mouthAperture(first) : null,
        onScreen: first ? onScreen(gazeFeatures(first)) : null, phone,
      });
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop() { cancelAnimationFrame(this.raf); }

  close() {
    this.stop();
    try {
      this.face.close();
      this.objects?.close();
    } catch (e) {
      console.warn("on-device vision close failed", e);
    }
  }
}

function gpuUsable(): boolean {
  try {
    const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : document.createElement("canvas");
    const gl = c.getContext("webgl2") as WebGL2RenderingContext | null;
    if (!gl) return false;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)).toLowerCase() : "";
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return !/swiftshader|llvmpipe|software/.test(renderer);
  } catch {
    return false;
  }
}

type HTTPVideoLike = HTMLVideoElement;

/** Square face crop around the landmarks, as base64 JPEG (no data: prefix). */
export function faceCrop(video: HTMLVideoElement, lm: Pt[], size = 160): string | null {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h || !lm.length) return null;
  const xs = lm.map((p) => p.x * w), ys = lm.map((p) => p.y * h);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const side = Math.max(x1 - x0, y1 - y0) * 1.3;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  c.getContext("2d")!.drawImage(video, cx - side / 2, cy - side / 2, side, side, 0, 0, size, size);
  return c.toDataURL("image/jpeg", 0.85).split(",")[1] ?? null;
}

export const roundLandmarks = (lm: Pt[]) => lm.map((p) => [Math.round(p.x * 1e4) / 1e4, Math.round(p.y * 1e4) / 1e4]);
