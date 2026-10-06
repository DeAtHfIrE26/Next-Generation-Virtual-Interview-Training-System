"""Interviewer personas: a name, a speaking style, a voice per TTS provider and an avatar look.

The look is applied by the web client to the 3D character (docs/DECISIONS.md D4); voices map to
:mod:`interview_core.realtime.tts` voice ids. All personas are female-presenting because the one
licensed avatar model is (D6); male voices stay in the voice tables for when a second model lands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    title: str
    style: str  # one line added to the interviewer prompt
    kokoro_voice: str
    polly_voice: str
    look: dict  # colours/variant consumed by the web avatar

    def public(self) -> dict:
        d = asdict(self)
        d.pop("style")
        return d


PERSONAS: dict[str, Persona] = {
    p.id: p
    for p in (
        Persona(
            "maya",
            "Maya",
            "Engineering Manager",
            "Warm, encouraging and precise; probes for ownership and measurable impact.",
            "maya",
            "ruth",
            {"skin": "#c99a7a", "hair": "#2b1d16", "top": "#1f2a44", "accent": "#7c9cff"},
        ),
        Persona(
            "emma",
            "Emma",
            "Principal Engineer",
            "Calm, technical and direct; pushes on trade-offs, failure modes and depth.",
            "emma",
            "amy",
            {"skin": "#e0b8a0", "hair": "#4a3426", "top": "#2d2f33", "accent": "#5ad1b5"},
        ),
        Persona(
            "priya",
            "Priya",
            "Talent Partner",
            "Friendly and structured; focuses on behaviour, motivation and clear examples.",
            "priya",
            "kajal",
            {"skin": "#a8765a", "hair": "#120c0a", "top": "#5b2a3c", "accent": "#ffb86b"},
        ),
        Persona(
            "ananya",
            "Ananya",
            "Staff Data Scientist",
            "Curious and rigorous; asks how results were measured and validated.",
            "ananya",
            "kajal",
            {"skin": "#9c6b4e", "hair": "#16100c", "top": "#20363a", "accent": "#8fd16a"},
        ),
    )
}


def persona(pid: str | None) -> Persona:
    return PERSONAS.get(pid or "", PERSONAS["maya"])
