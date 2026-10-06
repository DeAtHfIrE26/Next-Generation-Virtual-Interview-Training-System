"""Local speech models: pinned download URLs, checksums and licences.

All local models come from the sherpa-onnx GitHub releases (Apache-2.0 project). They are
downloaded once into ``MODELS_DIR`` (default ``~/.cache/interview-coach/models``) and verified
against the pinned SHA-256 before extraction. ``python -m interview_core.realtime.assets`` fetches
everything the configured providers need.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    sha256: str
    licence: str
    kind: str  # "tar.bz2" | "file"
    dirname: str  # directory (or file) name after extraction

    def path(self, root: Path | None = None) -> Path:
        return (root or models_dir()) / self.dirname


ASSETS: dict[str, Asset] = {
    a.key: a
    for a in (
        Asset(
            "stt-streaming",
            f"{BASE}/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
            "78e2b79fcf7271553a74402a76b771b09ea40117a39566a79f52235b23db6358",
            "NVIDIA Open Model License (upstream model card)",
            "tar.bz2",
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        ),
        Asset(
            "stt-final",
            f"{BASE}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
            "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
            "CC-BY-4.0 (upstream model card): attribution required",
            "tar.bz2",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        ),
        Asset(
            "vad",
            f"{BASE}/asr-models/silero_vad.onnx",
            "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
            "MIT (Silero VAD)",
            "file",
            "silero_vad.onnx",
        ),
        Asset(
            "tts-kokoro",
            f"{BASE}/tts-models/kokoro-multi-lang-v1_0.tar.bz2",
            "c5f7e2d2caf082bc1d20fb70334a61d99d20b484500aad32e7cf84c128ea3298",
            "Apache-2.0 (Kokoro-82M); phonemiser espeak-ng is GPL-3.0",
            "tar.bz2",
            "kokoro-multi-lang-v1_0",
        ),
    )
}


def models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", Path.home() / ".cache" / "interview-coach" / "models"))


def is_present(key: str, root: Path | None = None) -> bool:
    return ASSETS[key].path(root).exists()


def require(key: str, root: Path | None = None) -> Path:
    """Path of an installed asset, or a clear error telling the operator how to install it."""
    p = ASSETS[key].path(root)
    if not p.exists():
        raise FileNotFoundError(
            f"speech model '{key}' is not installed at {p}. "
            "Run: python -m interview_core.realtime.assets download"
        )
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(key: str, root: Path | None = None, *, verify: bool = True) -> Path:
    a = ASSETS[key]
    root = root or models_dir()
    dest = a.path(root)
    if dest.exists():
        return dest
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp_file = Path(tmp) / "download"
        with urllib.request.urlopen(a.url, timeout=60) as r, tmp_file.open("wb") as f:  # noqa: S310 - pinned https URL
            shutil.copyfileobj(r, f, length=1 << 20)
        digest = _sha256(tmp_file)
        if verify and digest != a.sha256:
            raise RuntimeError(f"checksum mismatch for {key}: expected {a.sha256}, got {digest}")
        if a.kind == "file":
            shutil.move(str(tmp_file), dest)
        else:
            with tarfile.open(tmp_file, "r:bz2") as tf:
                tf.extractall(tmp, filter="data")
            shutil.move(str(Path(tmp) / a.dirname), dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download and verify local speech models")
    d.add_argument("keys", nargs="*", default=list(ASSETS), help=f"subset of {list(ASSETS)}")
    sub.add_parser("list", help="show installed models")
    h = sub.add_parser("hash", help="print the SHA-256 of a downloaded file (for pinning)")
    h.add_argument("file")
    args = ap.parse_args(argv)
    if args.cmd == "hash":
        print(_sha256(Path(args.file)))
        return 0
    if args.cmd == "list":
        for k, a in ASSETS.items():
            print(f"{k:14s} {'installed' if a.path().exists() else 'missing':10s} {a.licence}")
        return 0
    for k in args.keys:
        print(f"{k}: {download(k)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
