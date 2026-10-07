"""ONNX face-embedding backend (ArcFace-style 112x112 input).

Bring your own commercially licensed ONNX model (for example a licensed InsightFace model
or one trained on commercially licensed data). The prototype's ``buffalo_l`` weights are
non-commercial and must not be used here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxFaceEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 112,
        mean: float = 127.5,
        std: float = 128.0,
        model_id: str | None = None,
    ):
        import onnxruntime as ort  # optional extra: interview-core[onnx]

        self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self.size, self.mean, self.std = input_size, mean, std
        self.model_id = model_id or f"onnx:{Path(model_path).name}"

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        img = _resize_nearest(face_rgb, self.size)
        x = ((img.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)[None]
        return np.asarray(self._sess.run(None, {self._input: x})[0]).ravel()


def _resize_nearest(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return img[ys][:, xs]
