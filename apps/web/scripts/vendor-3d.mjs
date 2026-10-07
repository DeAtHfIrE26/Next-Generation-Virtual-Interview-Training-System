// Copies the browser-side 3D and voice runtimes into public/vendor so they load from our own origin
// as native ES modules (TalkingHead imports lip-sync modules by computed path, which bundlers
// cannot follow). Runs on postinstall. Everything copied is MIT / ISC / Apache-2.0 licensed.
//
//   public/vendor/three/        three.module.js, three.core.js and only the addons TalkingHead imports
//   public/vendor/talkinghead/  TalkingHead 1.7.0 modules (+ one patch: meshopt decoder, see below)
//   public/vendor/headaudio/    HeadAudio worklet + viseme model (audio-driven lip-sync)
//   public/vendor/vad/          Silero VAD worklet + model (@ricky0123/vad-web)
//   public/vendor/ort/          onnxruntime-web WASM runtime used by the VAD
import { copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.join(here, "..");
const nm = path.join(web, "node_modules");
const out = path.join(web, "public", "vendor");

async function cp(src, dest) {
  await mkdir(path.dirname(dest), { recursive: true });
  await copyFile(src, dest);
}

// ---- three.js: core + the transitive closure of the addons TalkingHead and our code import
const threeRoot = path.join(nm, "three");
await cp(path.join(threeRoot, "build/three.module.js"), path.join(out, "three/three.module.js"));
await cp(path.join(threeRoot, "build/three.core.js"), path.join(out, "three/three.core.js"));
const addonRoot = path.join(threeRoot, "examples/jsm");
const entries = [
  "controls/OrbitControls.js",
  "loaders/GLTFLoader.js",
  "loaders/DRACOLoader.js",
  "loaders/FBXLoader.js",
  "environments/RoomEnvironment.js",
  "libs/stats.module.js",
  "libs/meshopt_decoder.module.js",
];
const seen = new Set();
async function addon(rel) {
  if (seen.has(rel)) return;
  seen.add(rel);
  const src = path.join(addonRoot, rel);
  await cp(src, path.join(out, "three/addons", rel));
  const code = await readFile(src, "utf8");
  for (const m of code.matchAll(/(?:import|export)[^'"]*?from\s*['"](\.{1,2}\/[^'"]+)['"]/g)) {
    await addon(path.posix.normalize(path.posix.join(path.posix.dirname(rel), m[1])));
  }
}
for (const e of entries) await addon(e);

// ---- TalkingHead (MIT). Patch: enable EXT_meshopt_compression so the avatar ships at 4 MB instead
// of 37 MB. The anchor must exist; fail loudly if a TalkingHead upgrade changes it.
const thSrc = path.join(nm, "@met4citizen/talkinghead/modules");
for (const f of await readdir(thSrc)) await cp(path.join(thSrc, f), path.join(out, "talkinghead", f));
const thFile = path.join(out, "talkinghead/talkinghead.mjs");
let th = await readFile(thFile, "utf8");
const anchor = "const loader = new GLTFLoader();";
if (!th.includes(anchor)) throw new Error("TalkingHead patch anchor not found; review scripts/vendor-3d.mjs");
th = th
  .replace(
    "import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';",
    "import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';\nimport { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js'; // patched: meshopt",
  )
  .replace(anchor, `${anchor}\n    loader.setMeshoptDecoder(MeshoptDecoder); // patched: meshopt`);
await writeFile(thFile, th);

// ---- HeadAudio (MIT): audio-driven viseme detection from the audio actually playing
const ha = path.join(nm, "@met4citizen/headaudio/dist");
for (const f of ["headaudio.min.mjs", "headworklet.min.mjs", "model-en-mixed.bin"]) {
  await cp(path.join(ha, f), path.join(out, "headaudio", f));
}

// ---- Silero VAD (ISC wrapper, MIT model) + onnxruntime-web WASM (MIT)
const vad = path.join(nm, "@ricky0123/vad-web/dist");
for (const f of await readdir(vad)) {
  if (f === "silero_vad_v5.onnx" || f === "vad.worklet.bundle.min.js") await cp(path.join(vad, f), path.join(out, "vad", f));
}
const ort = path.join(nm, "onnxruntime-web/dist");
for (const f of await readdir(ort)) {
  if (/^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f)) await cp(path.join(ort, f), path.join(out, "ort", f));
}

if (!existsSync(path.join(out, "vad/silero_vad_v5.onnx"))) throw new Error("VAD model missing from @ricky0123/vad-web");
console.log(`vendor-3d: ${seen.size} three addons, TalkingHead (patched), HeadAudio, VAD, ORT -> public/vendor`);
