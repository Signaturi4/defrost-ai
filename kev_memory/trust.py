"""Documentation trust: how far Claude may rely on a project's docs before reading its code.

    high  legacy / well-documented project: answer from the doc sections; read code only where a hit carries a `!`
          line (file changed after the doc, or the doc names code that no longer exists).
    low   fast-changing code, few docs (default): the sections are hints; read the `verify in:` files and answer from
          the code, citing the doc only where the code agrees.

Stored per domain as "doc_trust" in ~/.kev-memory/<domain>.workspace.json (`kev-memory setup --doc-trust`).
Default "low": a wrong answer copied from a stale doc is silent and costly, while grounding costs a few file reads."""
from __future__ import annotations

import json
from pathlib import Path

LEVELS = ("high", "low")
DEFAULT = "low"

HEADER = {
    "high": "doc trust: HIGH (well-documented project). Answer from these sections; read code only for hits with a "
            "`!` line; if the code disagrees, ask the user which is right (see conflicts.QUESTION).",
    "low": "doc trust: LOW (code changes fast, docs lag). Treat these sections as hints: read the `verify in:` files "
           "and answer from the code; cite a doc only where the code agrees. Doc/code disagreements: ask the user.",
}

RULE = {
    "high": "- **Ground (doc trust: high):** this project's docs are maintained, so answer from the returned sections. "
            "Read code only when a hit has a line starting with `!` (file changed after the doc, or the doc names "
            "code that no longer exists). If the code disagrees with the doc, the user decides which is right: "
            "see the conflict rule below.",
    "low": "- **Ground (doc trust: low):** this project's code moves faster than its docs, so the sections are hints. "
           "Before stating how something behaves, read the files on the hit's `verify in:` line (and always those "
           "behind a `!` line) and answer from what the code does. When code and doc disagree, the user decides which "
           "is right: see the conflict rule below.",
}


def read(workspace: str | Path | None) -> str:
    try:
        level = json.loads(Path(workspace).read_text()).get("doc_trust") if workspace else None
    except (OSError, ValueError):
        level = None
    return level if level in LEVELS else DEFAULT


def write(workspace: str | Path, level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"doc trust must be one of {LEVELS}, not {level!r}")
    p = Path(workspace)
    spec = json.loads(p.read_text())
    spec["doc_trust"] = level
    p.write_text(json.dumps(spec, indent=1))
