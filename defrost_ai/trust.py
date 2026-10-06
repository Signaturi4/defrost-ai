"""Documentation trust: how far Claude may rely on a project's docs before reading its code.

    high  legacy / well-documented project: answer from the doc sections; read code only where a hit carries a `!`
          line (file changed after the doc, or the doc names code that no longer exists).
    low   fast-changing code, few docs (default): the sections are hints; read the `verify in:` files and answer from
          the code, citing the doc only where the code agrees.

Stored per domain as "doc_trust" in ~/.defrost-ai/<domain>.workspace.json (`defrost setup --doc-trust`).
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
           "and answer from the code; cite a doc only where the code agrees. Doc/code disagreements you verified in the code: state both versions in the answer, then ask the user which is right (at most 2 questions).",
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


def stored(workspace: str | Path | None) -> str | None:
    """The level chosen for this project, or None when none was ever chosen (read() then falls back to DEFAULT)."""
    try:
        level = json.loads(Path(workspace).read_text()).get("doc_trust") if workspace else None
    except (OSError, ValueError):
        return None
    return level if level in LEVELS else None


CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
            ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".scala", ".sh", ".sql", ".vue", ".svelte", ".m", ".mm", ".dart",
            ".lua", ".ex", ".exs", ".tf"}
DOC_EXT = {".md", ".mdx", ".rst", ".txt", ".adoc"}
KIT = ("docs/tools/", "docs/templates/")                    # the doc-rules kit setup copies in: not the project's code


def suggest(root: str | Path) -> tuple[str, str]:
    """Recommended level for a repository, with the reason: "high" for a docs repository (at least 10 doc files per
    code file: the docs are what there is to know, a few helper scripts aside), else "low". Hidden folders (.claude,
    .agents, ...) are tooling, not the project's code."""
    import subprocess
    root = Path(root)
    r = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    files = r.stdout.splitlines() if r.returncode == 0 and r.stdout.strip() else \
        [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
    files = [f for f in files if not any(part.startswith(".") for part in Path(f).parts) and not f.startswith(KIT)]
    code = sum(Path(f).suffix.lower() in CODE_EXT for f in files)
    docs = sum(Path(f).suffix.lower() in DOC_EXT for f in files)
    if docs and docs >= 10 * code:
        return "high", (f"{docs} doc files and {code} code files: a docs repository, the docs are the source of truth"
                        if code else f"{docs} doc files and no code: the docs are the source of truth")
    return "low", f"{code} code files and {docs} doc files: code changes faster than docs"
