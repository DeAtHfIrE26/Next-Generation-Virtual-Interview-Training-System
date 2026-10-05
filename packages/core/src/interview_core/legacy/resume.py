"""Prototype resume handling (E6 parse).

Source: ``legacy/desktop/main.py`` ``extract_candidate_name`` and ``build_context``
(L882-904). PDF text extraction used PyMuPDF (AGPL); the product uses pypdf instead,
see :mod:`interview_core.nlp.resume`.
"""

from __future__ import annotations

import re


def extract_candidate_name(resume_text: str) -> str:
    lines = [ln.strip() for ln in resume_text.split("\n") if ln.strip()]
    if lines:
        first = lines[0]
        if re.match(r"^[A-Za-z\s]+$", first) and len(first.split()) <= 4:
            return first.strip()
    return "Candidate"


def build_context(resume_text: str, role: str) -> str:
    length = 600
    summary = resume_text[:length].replace("\n", " ")
    if len(resume_text) > length:
        summary += "..."
    name = extract_candidate_name(resume_text)
    return (
        f"Candidate's Desired Role: {role}\n"
        f"Resume Summary: {summary}\n"
        f"Candidate Name: {name}\n\n"
        "You are a seasoned interviewer. Ask professional role-based or scenario-based questions."
    )
