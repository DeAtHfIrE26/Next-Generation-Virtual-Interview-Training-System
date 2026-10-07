"""Role family and seniority helpers."""

from __future__ import annotations

import re
from typing import Literal

Family = Literal["software", "data", "product", "design", "business", "general"]
Seniority = Literal["intern", "junior", "mid", "senior", "lead"]

_FAMILY_RULES: list[tuple[Family, tuple[str, ...]]] = [
    ("data", ("data", "analyst", "analytics", "machine learning", "ml ", "scientist", "bi ")),
    (
        "software",
        (
            "software",
            "developer",
            "engineer",
            "programmer",
            "devops",
            "sde",
            "backend",
            "frontend",
            "full stack",
            "fullstack",
            "sre",
            "mobile",
            "qa",
            "test",
        ),
    ),
    ("product", ("product manager", "product owner", "pm", "program manager")),
    ("design", ("designer", "ux", "ui ", "user experience", "researcher")),
    (
        "business",
        (
            "sales",
            "marketing",
            "operations",
            "hr",
            "human resources",
            "finance",
            "accountant",
            "consultant",
            "business",
            "recruiter",
            "customer success",
        ),
    ),
]

START_DIFFICULTY: dict[Seniority, int] = {"intern": 1, "junior": 2, "mid": 3, "senior": 4, "lead": 4}


def role_family(role: str) -> Family:
    r = f" {role.lower()} "
    for fam, keys in _FAMILY_RULES:
        if any(re.search(rf"(?<![a-z]){re.escape(k.strip())}(?![a-z])", r) for k in keys):
            return fam
    return "general"


def is_technical(family: Family) -> bool:
    return family in ("software", "data")
