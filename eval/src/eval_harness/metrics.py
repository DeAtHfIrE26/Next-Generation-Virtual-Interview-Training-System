"""Evaluation metrics. Pure numpy, each verified against hand-computed values in tests."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- ASR


@dataclass(frozen=True)
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int


_NORMALISE = re.compile(r"[^\w\s']")


def normalise_text(text: str) -> list[str]:
    """Lower-case, strip punctuation except apostrophes, split on whitespace."""
    return _NORMALISE.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Levenshtein word alignment; WER = (S + D + I) / N."""
    ref, hyp = normalise_text(reference), normalise_text(hypothesis)
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, S, D, I) aligning ref[:i] with hyp[:j]
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            c, s, d, ins = dp[i - 1][j - 1]
            sub = (c + 1, s + 1, d, ins)
            c, s, d, ins = dp[i - 1][j]
            dele = (c + 1, s, d + 1, ins)
            c, s, d, ins = dp[i][j - 1]
            inse = (c + 1, s, d, ins + 1)
            dp[i][j] = min(sub, dele, inse, key=lambda t: t[0])
    cost, s, d, ins = dp[n][m]
    wer = cost / n if n else (0.0 if m == 0 else float("inf"))
    return WerResult(wer, s, d, ins, n)


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> WerResult:
    """Corpus WER = total edits / total reference words (not the mean of per-utterance WERs)."""
    s = d = i = n = 0
    for ref, hyp in pairs:
        r = word_error_rate(ref, hyp)
        s, d, i, n = s + r.substitutions, d + r.deletions, i + r.insertions, n + r.reference_words
    return WerResult((s + d + i) / n if n else 0.0, s, d, i, n)


# ---------------------------------------------------------------- biometric verification


@dataclass(frozen=True)
class VerificationResult:
    eer: float
    eer_threshold: float
    far_at_threshold: float | None
    frr_at_threshold: float | None
    threshold: float | None
    genuine: int
    impostor: int


def far_frr(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    """Higher score = more similar; accept when score >= threshold."""
    far = float(np.mean(impostor >= threshold)) if impostor.size else 0.0
    frr = float(np.mean(genuine < threshold)) if genuine.size else 0.0
    return far, frr


def equal_error_rate(genuine: Sequence[float], impostor: Sequence[float]) -> tuple[float, float]:
    """Return ``(eer, threshold)`` by sweeping every observed score as a threshold.

    EER is reported as the mean of FAR and FRR at the threshold where they are closest.
    """
    g, imp = np.asarray(genuine, float), np.asarray(impostor, float)
    if g.size == 0 or imp.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    thresholds = np.unique(np.concatenate([g, imp, [np.inf]]))
    best = (2.0, 0.0, 0.0)
    for t in thresholds:
        far, frr = far_frr(g, imp, t)
        gap = abs(far - frr)
        if gap < best[0]:
            best = (gap, (far + frr) / 2, float(t))
    return best[1], best[2]


def verification_report(
    genuine: Sequence[float], impostor: Sequence[float], threshold: float | None = None
) -> VerificationResult:
    eer, eer_t = equal_error_rate(genuine, impostor)
    far = frr = None
    if threshold is not None:
        far, frr = far_frr(np.asarray(genuine, float), np.asarray(impostor, float), threshold)
    return VerificationResult(eer, eer_t, far, frr, threshold, len(genuine), len(impostor))


def threshold_for_far(impostor: Sequence[float], target_far: float) -> float:
    """Smallest threshold whose FAR on these impostor scores is <= target_far."""
    imp = np.sort(np.asarray(impostor, float))
    if imp.size == 0:
        raise ValueError("no impostor scores")
    for t in np.unique(np.concatenate([imp, [np.inf]])):
        if float(np.mean(imp >= t)) <= target_far:
            return float(t)
    return float("inf")


def apcer_bpcer(attack_scores: Sequence[float], bonafide_scores: Sequence[float], threshold: float):
    """ISO/IEC 30107-3 style rates for a liveness score where higher = more likely live.

    APCER: attacks accepted as live. BPCER: bona fide rejected.
    """
    a, b = np.asarray(attack_scores, float), np.asarray(bonafide_scores, float)
    apcer = float(np.mean(a >= threshold)) if a.size else 0.0
    bpcer = float(np.mean(b < threshold)) if b.size else 0.0
    return apcer, bpcer


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """AUC = P(score_pos > score_neg) + 0.5 * P(tie) (Mann-Whitney U)."""
    p, n = np.asarray(positive, float), np.asarray(negative, float)
    if p.size == 0 or n.size == 0:
        raise ValueError("need positives and negatives")
    ranks = _average_ranks(np.concatenate([p, n]))
    u = ranks[: p.size].sum() - p.size * (p.size + 1) / 2
    return float(u / (p.size * n.size))


# ---------------------------------------------------------------- agreement


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.size != y.size or x.size < 2:
        raise ValueError("need two equal-length sequences of length >= 2")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def cohen_kappa(a: Sequence[int], b: Sequence[int], *, weights: str | None = None) -> float:
    """Cohen's kappa for two raters; ``weights='quadratic'`` gives QWK for ordinal scales."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.size != b_arr.size or a_arr.size == 0:
        raise ValueError("need two equal-length, non-empty sequences")
    cats = np.unique(np.concatenate([a_arr, b_arr]))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a_arr, b_arr, strict=True):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights is None:
        w = 1 - np.eye(k)
    elif weights == "quadratic":
        grid = np.arange(k)
        w = (grid[:, None] - grid[None, :]) ** 2 / max((k - 1) ** 2, 1)
    else:
        raise ValueError("weights must be None or 'quadratic'")
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else 1.0


# ---------------------------------------------------------------- detection and latency


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    x = np.asarray(samples_ms, float)
    if x.size == 0:
        raise ValueError("no samples")
    return {
        "n": int(x.size),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "max_ms": float(x.max()),
    }


def bootstrap_ci(
    values: Sequence[float], stat=np.mean, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
):
    """Percentile bootstrap confidence interval for ``stat`` over ``values``."""
    x = np.asarray(values, float)
    if x.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))
