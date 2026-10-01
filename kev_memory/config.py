"""Workspace configuration: which folders feed one memory, and where the memory is written.

A workspace is a JSON file:

    {"name": "acme", "out": "~/.kev-memory/acme",
     "components": [{"name": "backend", "path": "~/src/api", "exclude": ["vendor"], "role": "REST API"},
                    {"name": "handbook", "path": "~/docs/handbook", "text_only": true}]}

A component may list several roots under "paths". Nothing is ever written inside a component; all output goes to
`out`. `text_only` components contribute documents but no code graph."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".history", "graphify-out", ".terraform", "dist", "build", ".next", "Pods",
             "coverage", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".expo"}
DOC_SUFFIXES = {".md", ".mdx", ".rst", ".asc", ".adoc"}
CODE_SUFFIXES = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java", ".kt", ".kts", ".swift", ".go", ".rs",
                 ".rb", ".php", ".cs", ".scala", ".c", ".h", ".cpp", ".hpp", ".m", ".dart", ".lua"}
ICLOUD_PLACEHOLDER = 0x40000000   # macOS SF_DATALESS: file content not on disk; reading it would block on a download


@dataclass
class Component:
    name: str
    roots: list[Path]
    exclude: set[str] = field(default_factory=set)
    role: str = ""
    text_only: bool = False
    skipped: list[Path] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: dict) -> "Component":
        roots = [Path(os.path.expanduser(p)).resolve() for p in spec.get("paths", [spec.get("path")]) if p]
        return cls(spec["name"], roots, set(spec.get("exclude", [])), spec.get("role", ""), spec.get("text_only", False))

    def files(self, suffixes: set[str]):
        """Yield (root, path) for files with one of `suffixes`, skipping SKIP_DIRS, hidden dirs and excludes."""
        for root in self.roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                rel = Path(dirpath).relative_to(root)
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and (not d.startswith(".") or d == ".github")
                                     and str(rel / d) not in self.exclude and d not in self.exclude)
                for name in sorted(filenames):
                    path = Path(dirpath) / name
                    if path.suffix.lower() not in suffixes:
                        continue
                    if getattr(os.stat(path), "st_flags", 0) & ICLOUD_PLACEHOLDER:
                        if path not in self.skipped:
                            self.skipped.append(path)
                        continue
                    yield root, path

    def display_path(self, root: Path, path: Path) -> str:
        """Stable path shown in citations: '<component>/<path inside root>' (plus the root name if there are several)."""
        inner = path.relative_to(root).as_posix()
        return f"{self.name}/{root.name}/{inner}" if len(self.roots) > 1 else f"{self.name}/{inner}"


@dataclass
class Workspace:
    name: str
    out: Path
    components: list[Component]
    source: Path | None = None

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Workspace":
        spec = json.loads(Path(path).read_text())
        out = Path(os.path.expanduser(spec.get("out", f"~/.kev-memory/{spec['name']}")))
        return cls(spec["name"], out, [Component.from_spec(c) for c in spec["components"]], Path(path).resolve())


def git_times(root: Path, paths: list[Path]) -> dict[Path, int]:
    """{path: last commit unix time}; files outside git fall back to mtime. One `git log` for the whole repo."""
    seen = {}
    try:
        log = subprocess.run(["git", "-C", str(root), "log", "--format=%x01%ct", "--name-only"], capture_output=True,
                             text=True, timeout=120).stdout
        t = None
        for line in log.splitlines():
            if line.startswith("\x01"):
                t = int(line[1:])
            elif line and t is not None:
                seen.setdefault((root / line).resolve(), t)
    except (OSError, subprocess.SubprocessError):
        pass
    return {p: seen.get(p.resolve(), int(p.stat().st_mtime)) for p in paths}


def git_head(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
