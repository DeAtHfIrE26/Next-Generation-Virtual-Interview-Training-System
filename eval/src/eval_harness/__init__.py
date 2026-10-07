"""Evaluation harness. Run ``python -m eval_harness run --suite all``.

Accuracy numbers are produced only from real, consented data described by a manifest
under ``eval/data/<suite>/manifest.jsonl`` (never committed). The ``smoke`` suite uses
synthetic data to check that pipelines run; its outputs are never reported as accuracy.
"""
