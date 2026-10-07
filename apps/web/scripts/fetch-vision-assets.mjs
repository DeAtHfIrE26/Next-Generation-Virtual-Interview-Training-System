// Copies MediaPipe WASM from node_modules and downloads the two Apache-2.0 vision models into
// public/vision so the browser loads them from our own origin (no third-party CDN at runtime).
// Skips quietly when offline; the app then reports vision features as unavailable.
import { copyFile, mkdir, readdir, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

const root = path.dirname(new URL(import.meta.url).pathname);
const out = path.join(root, "..", "public", "vision");
const wasmSrc = path.join(root, "..", "node_modules", "@mediapipe", "tasks-vision", "wasm");
const MODELS = {
  "face_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  "efficientdet_lite0.tflite":
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
};

await mkdir(path.join(out, "wasm"), { recursive: true });
if (existsSync(wasmSrc)) {
  for (const f of await readdir(wasmSrc)) await copyFile(path.join(wasmSrc, f), path.join(out, "wasm", f));
}
for (const [name, url] of Object.entries(MODELS)) {
  const dest = path.join(out, name);
  if (existsSync(dest) && (await stat(dest)).size > 0) continue;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await writeFile(dest, Buffer.from(await res.arrayBuffer()));
    console.log(`vision: downloaded ${name}`);
  } catch (e) {
    console.warn(`vision: could not download ${name} (${e.message}); vision features will be unavailable`);
  }
}
