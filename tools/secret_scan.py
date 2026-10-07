"""Fail if the working tree contains likely credentials.

Prints only file:line and the rule name, never the matched value.
Usage: python tools/secret_scan.py [paths...]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

RULES: dict[str, re.Pattern[str]] = {
    "rapidapi_key": re.compile(r"\b[0-9a-f]{10}msh[0-9a-f]{15}p[0-9a-f]{6}jsn[0-9a-f]{12}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "stripe_key": re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    "razorpay_key": re.compile(r"\brzp_(?:live|test)_[A-Za-z0-9]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Generic: a quoted 24+ char token assigned to something named like a key/secret/token.
    "assigned_secret": re.compile(
        r"""(?i)(?:api[_-]?key|secret|token|password)["']?\s*[:=]\s*(?:os\.getenv\([^,]+,\s*)?["'][A-Za-z0-9_\-]{24,}["']"""
    ),
}

PLACEHOLDER = re.compile(r"(?i)your_|_here|placeholder|example|xxxx|changeme|dummy|redacted")

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".pt",
    ".caffemodel",
    ".onnx",
    ".wav",
    ".ico",
    ".woff2",
}


def tracked_files(paths: list[str]) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", *paths],
        capture_output=True,
        check=True,
    ).stdout
    return [Path(p) for p in out.decode().split("\0") if p]


def main(argv: list[str]) -> int:
    findings = 0
    for path in tracked_files(argv):
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, rule in RULES.items():
                match = rule.search(line)
                if match and not PLACEHOLDER.search(match.group(0)):
                    print(f"{path}:{lineno}: possible {name}")
                    findings += 1
    if findings:
        print(f"secret_scan: {findings} finding(s). Values not printed.")
        return 1
    print("secret_scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
