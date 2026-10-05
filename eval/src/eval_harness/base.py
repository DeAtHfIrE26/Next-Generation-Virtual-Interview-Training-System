"""Shared types for evaluation suites."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["measured", "no_data", "not_configured", "smoke", "error"]

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass
class SystemResult:
    """Metrics for one system (algorithm version) on one suite."""

    system: str
    metrics: dict[str, Any]
    subgroups: dict[str, dict[str, Any]] = field(default_factory=dict)
    n: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class SuiteResult:
    suite: str
    status: Status
    description: str
    data_needed: str
    systems: list[SystemResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ManifestError(ValueError):
    pass


REQUIRED_COMMON = ("id", "consent_id")


def load_manifest(path: Path, required: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Read a JSONL manifest. Every item must reference a consent record.

    Paths inside items are resolved relative to the manifest's directory.
    """
    items: list[dict[str, Any]] = []
    need = (*REQUIRED_COMMON, *required)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ManifestError(f"{path}:{lineno}: invalid JSON ({e.msg})") from e
        missing = [k for k in need if item.get(k) in (None, "")]
        if missing:
            raise ManifestError(f"{path}:{lineno}: missing required field(s) {missing}")
        item["_base"] = str(path.parent)
        items.append(item)
    if not items:
        raise ManifestError(f"{path}: empty manifest")
    return items


def resolve(item: dict[str, Any], key: str) -> Path:
    return Path(item["_base"]) / item[key]


def group_by(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group items by ``item['subgroups'][key]``; items without that attribute are skipped."""
    out: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        val = (it.get("subgroups") or {}).get(key)
        if val not in (None, ""):
            out.setdefault(str(val), []).append(it)
    return out


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    data_needed: str
    run: Callable[[Path, dict[str, Any]], SuiteResult]


def manifest_path(data_root: Path, suite: str) -> Path:
    return data_root / suite / "manifest.jsonl"


def _display(path: Path) -> str:
    """Repo-relative path for reports, so local absolute paths never end up in committed files."""
    try:
        return str(path.resolve().relative_to(DEFAULT_DATA_ROOT.parents[1]))
    except ValueError:
        return f"<data-root>/{path.parent.name}/{path.name}"


def no_data(suite: Suite, data_root: Path) -> SuiteResult:
    return SuiteResult(
        suite.name,
        "no_data",
        suite.description,
        suite.data_needed,
        notes=[f"No manifest at {_display(manifest_path(data_root, suite.name))}. Not measured."],
    )
