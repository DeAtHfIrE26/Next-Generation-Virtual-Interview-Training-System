"""STT benchmark (docs/DECISIONS.md D1). Synthetic interview answers spoken by Kokoro voices.

Env: BENCH_DIR (audio + manifest.json, default eval/benchmarks/out), BENCH_MODELS (sherpa-onnx model dirs).
Results of the run cited in DECISIONS.md: docs/evidence/voice/stt_bench.txt
"""

import os

BENCH_DIR = os.environ.get("BENCH_DIR", "eval/benchmarks/out")
BENCH_MODELS = os.environ.get("BENCH_MODELS", ".models/bench")
import json
import re
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

man = json.load(open(os.path.join(BENCH_DIR, "manifest.json")))
M = BENCH_MODELS
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def numword(n):
    n = int(n)
    if n < 20:
        return ONES[n]
    if n in TENS:
        return TENS[n]
    return TENS[n // 10 * 10] + " " + ONES[n % 10]


def norm(s):
    s = s.lower().replace("%", " percent").replace("½", " half")
    s = re.sub(r"\d+", lambda m: numword(m.group()) if int(m.group()) < 60 else m.group(), s)
    s = re.sub(r"[^a-z' ]", " ", s)
    return s.split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def streaming(name, **kw):
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(num_threads=4, decoding_method="greedy_search", **kw)
    E = N = 0
    comp = audio = 0.0
    firsts = []
    per_voice = {}
    for m in man:
        x, sr = sf.read(os.path.join(BENCH_DIR, os.path.basename(m["path"])), dtype="float32")
        s = rec.create_stream()
        t0 = time.time()
        first = None
        for k in range(0, len(x), 1600):  # 100 ms chunks, as a browser would send
            s.accept_waveform(sr, x[k : k + 1600])
            while rec.is_ready(s):
                rec.decode_stream(s)
            if first is None and rec.get_result(s).strip():
                first = (k + 1600) / sr
        s.accept_waveform(sr, np.zeros(int(0.8 * sr), np.float32))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        hyp = rec.get_result(s)
        comp += time.time() - t0
        audio += len(x) / sr
        e, n = wer(m["text"], hyp)
        E += e
        N += n
        pv = per_voice.setdefault(m["voice"], [0, 0])
        pv[0] += e
        pv[1] += n
        if first is not None:
            firsts.append(first)
    print(
        f"{name:34s} WER {E / N * 100:5.1f}%  RTF {comp / audio:.3f}  first-partial median {np.median(firsts):.2f}s audio  per-voice "
        + " ".join(f"{v}:{a / b * 100:.0f}%" for v, (a, b) in per_voice.items())
    )
    print("   sample:", hyp[:140])


def offline(name, **kw):
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(num_threads=4, decoding_method="greedy_search", **kw)
    E = N = 0
    comp = audio = 0.0
    per_voice = {}
    for m in man:
        x, sr = sf.read(os.path.join(BENCH_DIR, os.path.basename(m["path"])), dtype="float32")
        t0 = time.time()
        s = rec.create_stream()
        s.accept_waveform(sr, x)
        rec.decode_stream(s)
        hyp = s.result.text
        comp += time.time() - t0
        audio += len(x) / sr
        e, n = wer(m["text"], hyp)
        E += e
        N += n
        pv = per_voice.setdefault(m["voice"], [0, 0])
        pv[0] += e
        pv[1] += n
    print(
        f"{name:34s} WER {E / N * 100:5.1f}%  RTF {comp / audio:.3f}  (offline final pass) per-voice "
        + " ".join(f"{v}:{a / b * 100:.0f}%" for v, (a, b) in per_voice.items())
    )
    print("   sample:", hyp[:140])


z = f"{M}/sherpa-onnx-streaming-zipformer-en-2023-06-26"
streaming(
    "zipformer-en-2023-06-26 (int8)",
    tokens=f"{z}/tokens.txt",
    encoder=f"{z}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder=f"{z}/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    joiner=f"{z}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
)
f = f"{M}/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms"
streaming(
    "nemo-fastconformer-streaming-80ms",
    tokens=f"{f}/tokens.txt",
    encoder=f"{f}/encoder.onnx",
    decoder=f"{f}/decoder.onnx",
    joiner=f"{f}/joiner.onnx",
)
n = f"{M}/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
streaming(
    "nemotron-streaming-0.6b-560ms int8",
    tokens=f"{n}/tokens.txt",
    encoder=f"{n}/encoder.int8.onnx",
    decoder=f"{n}/decoder.int8.onnx",
    joiner=f"{n}/joiner.int8.onnx",
)
p = f"{M}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
offline(
    "parakeet-tdt-0.6b-v2 int8",
    tokens=f"{p}/tokens.txt",
    encoder=f"{p}/encoder.int8.onnx",
    decoder=f"{p}/decoder.int8.onnx",
    joiner=f"{p}/joiner.int8.onnx",
    model_type="nemo_transducer",
)
