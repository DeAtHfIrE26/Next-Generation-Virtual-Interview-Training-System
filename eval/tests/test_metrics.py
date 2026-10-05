import math

import pytest
from eval_harness import metrics as m


def test_wer_counts_substitution_and_deletion():
    r = m.word_error_rate("The cat sat on the mat.", "the cat sit on mat")
    assert (r.substitutions, r.deletions, r.insertions, r.reference_words) == (1, 1, 0, 6)
    assert r.wer == pytest.approx(2 / 6)


def test_wer_insertions_and_empty_cases():
    assert m.word_error_rate("hello", "hello there").insertions == 1
    assert m.word_error_rate("", "").wer == 0.0
    assert math.isinf(m.word_error_rate("", "noise").wer)


def test_corpus_wer_weights_by_reference_length():
    r = m.corpus_wer([("a b c d", "a b c d"), ("x y", "x")])
    assert r.wer == pytest.approx(1 / 6)


def test_eer_hand_computed():
    eer, t = m.equal_error_rate([0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.65])
    assert eer == pytest.approx(0.25)
    assert t == pytest.approx(0.65)


def test_eer_perfect_separation_is_zero():
    eer, _ = m.equal_error_rate([0.9, 0.95], [0.1, 0.2])
    assert eer == 0.0


def test_far_frr_and_threshold_for_far():
    g, i = [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.65]
    rep = m.verification_report(g, i, threshold=0.7)
    assert (rep.far_at_threshold, rep.frr_at_threshold) == (0.0, 0.25)
    assert m.threshold_for_far(i, 0.0) > 0.65
    assert m.threshold_for_far(i, 0.25) == pytest.approx(0.65)


def test_apcer_bpcer():
    assert m.apcer_bpcer([0.1, 0.6, 0.2, 0.9], [0.8, 0.4], 0.5) == (0.5, 0.5)


def test_auc_hand_computed_and_ties():
    assert m.roc_auc([0.9, 0.8], [0.1, 0.85]) == pytest.approx(0.75)
    assert m.roc_auc([0.5], [0.5]) == pytest.approx(0.5)


def test_spearman_known_values():
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert m.spearman([1, 2, 3], [1, 1, 2]) == pytest.approx(1.5 / math.sqrt(3))


def test_cohen_kappa_textbook_example():
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert m.cohen_kappa(a, b) == pytest.approx(0.4)


def test_quadratic_kappa():
    assert m.cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4], weights="quadratic") == pytest.approx(1.0)
    near = m.cohen_kappa([1, 2, 3, 4, 5], [1, 2, 3, 5, 4], weights="quadratic")
    far = m.cohen_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], weights="quadratic")
    assert near > 0.8 and far < 0


def test_precision_recall_latency_bootstrap():
    assert m.precision_recall(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    s = m.latency_summary([100, 200, 300, 400, 500])
    assert s["p50_ms"] == 300 and s["n"] == 5
    lo, hi = m.bootstrap_ci([1.0] * 10)
    assert lo == hi == 1.0
