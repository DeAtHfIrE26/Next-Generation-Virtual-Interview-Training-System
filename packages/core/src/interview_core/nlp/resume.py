"""Resume parsing (E6), replacing the prototype's raw PyMuPDF text dump.

- Text extraction with pypdf (BSD licence; PyMuPDF is AGPL).
- Section detection, skills list, rough years of experience.
- Contact details (emails, phone numbers, URLs) are removed before any text is sent to an
  LLM provider: the interviewer does not need them, and minimising personal data is required
  by DPDP purpose limitation.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date

from pypdf import PdfReader

SECTION_HEADINGS = {
    "summary": ("summary", "profile", "objective", "about me"),
    "experience": ("experience", "work experience", "employment", "professional experience", "work history"),
    "education": ("education", "academics", "qualifications"),
    "skills": ("skills", "technical skills", "core skills", "competencies", "tools"),
    "projects": ("projects", "personal projects", "academic projects"),
    "certifications": ("certifications", "certificates", "licenses"),
    "achievements": ("achievements", "awards", "honors", "honours"),
}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.I)
YEAR_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)\b", re.I)
MAX_CHARS = 12000


@dataclass
class ResumeProfile:
    text: str  # redacted full text, truncated to MAX_CHARS
    sections: dict[str, str] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    years_experience: float | None = None
    truncated: bool = False

    def context_for_llm(self, limit: int = 4000) -> str:
        parts = []
        for key in ("summary", "experience", "projects", "skills", "education", "certifications"):
            if self.sections.get(key):
                parts.append(f"## {key.title()}\n{self.sections[key][:1500]}")
        body = "\n\n".join(parts) or self.text
        return body[:limit]


def extract_pdf_text(data: bytes, max_pages: int = 10) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages[:max_pages]
    return "\n".join((p.extract_text() or "") for p in pages)


def redact_contacts(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = URL.sub("[link]", text)
    return PHONE.sub(_phone_or_keep, text)


def _phone_or_keep(m: re.Match[str]) -> str:
    """Redact only digit runs that look like phone numbers (10-15 digits), never year ranges."""
    s = m.group(0)
    digits = sum(c.isdigit() for c in s)
    if YEAR_RANGE.fullmatch(s.strip()) or not 10 <= digits <= 15:
        return s
    return "[phone]"


def split_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    out: dict[str, list[str]] = {}
    current = "summary"
    for line in lines:
        key = _heading(line)
        if key:
            current = key
            continue
        out.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in out.items() if "\n".join(v).strip()}


def _heading(line: str) -> str | None:
    s = re.sub(r"[^a-z ]", "", line.strip().lower()).strip()
    if not s or len(s) > 30:
        return None
    for key, names in SECTION_HEADINGS.items():
        if s in names:
            return key
    return None


def parse_skills(section: str) -> list[str]:
    raw = re.split(r"[,\n;|•·]", section)
    seen, out = set(), []
    for s in raw:
        s = re.sub(r"^[\s\-*:]+|[\s.]+$", "", s)
        s = re.sub(r"^[A-Za-z ]{2,20}:\s*", "", s)  # "Languages: Python" -> "Python"
        if 1 < len(s) <= 40 and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:60]


def years_of_experience(experience: str, today: date | None = None) -> float | None:
    today = today or date.today()
    spans = []
    for start, end in YEAR_RANGE.findall(experience):
        e = today.year if not end[0].isdigit() else int(end)
        s = int(start)
        if 1970 <= s <= e <= today.year + 1:
            spans.append((s, e))
    if not spans:
        return None
    # Merge overlapping year ranges so concurrent roles are not double counted.
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return float(sum(e - s for s, e in merged))


def parse_resume_text(text: str) -> ResumeProfile:
    clean = redact_contacts(text)
    truncated = len(clean) > MAX_CHARS
    clean = clean[:MAX_CHARS]
    sections = split_sections(clean)
    return ResumeProfile(
        clean,
        sections,
        parse_skills(sections.get("skills", "")),
        years_of_experience(sections.get("experience", "")),
        truncated,
    )


def parse_resume_pdf(data: bytes) -> ResumeProfile:
    return parse_resume_text(extract_pdf_text(data))
