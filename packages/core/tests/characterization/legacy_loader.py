"""Load individual functions from the prototype source without importing it.

Importing ``legacy/desktop/main.py`` would open a Tk window, a camera and a microphone,
so we parse it, pull out only the requested top-level functions, and execute them in a
namespace whose globals the test controls.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
LEGACY_MAIN = REPO_ROOT / "legacy" / "desktop" / "main.py"


def load_functions(names: list[str], namespace: dict[str, Any], source: Path = LEGACY_MAIN) -> dict[str, Any]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = set(names)
    found: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        # Later definitions win, matching Python's own semantics for redefined functions.
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = node
    missing = wanted - found.keys()
    if missing:
        raise LookupError(f"not found in {source}: {sorted(missing)}")
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}
