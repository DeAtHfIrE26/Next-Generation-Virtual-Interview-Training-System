"""Speaker-embedding backends.

- ``speechbrain_ecapa``: ECAPA-TDNN trained on VoxCeleb (SpeechBrain, Apache-2.0 code; check
  the model card licence of the exact checkpoint before production use).
- ``resemblyzer``: GE2E encoder bundled with Resemblyzer (Apache-2.0).
"""

from __future__ import annotations

import numpy as np


class SpeechBrainEcapa:
    model_id = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, savedir: str = ".models/ecapa"):
        from speechbrain.inference.speaker import EncoderClassifier  # optional extra

        self._clf = EncoderClassifier.from_hparams(source=self.model_id, savedir=savedir)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if sr != 16000:
            audio = _resample_linear(audio, sr, 16000)
        with torch.no_grad():
            emb = self._clf.encode_batch(torch.from_numpy(np.asarray(audio, np.float32))[None])
        return emb.squeeze().cpu().numpy()


class ResemblyzerEncoder:
    model_id = "resemblyzer-ge2e"

    def __init__(self):
        from resemblyzer import VoiceEncoder  # optional extra

        self._enc = VoiceEncoder(verbose=False)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from resemblyzer import preprocess_wav

        return self._enc.embed_utterance(preprocess_wav(np.asarray(audio, np.float32), source_sr=sr))


def _resample_linear(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    n = round(len(x) * target / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
