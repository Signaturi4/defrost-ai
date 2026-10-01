"""A workspace = one project split over several repos/folders ("components"), each scanned into the same memory.

    {"name": "client", "out": "~/client-memory", "doc_trust": "auto" | 0.0-1.0,
     "components": [{"name": "backend", "path": "~/src/api", "exclude": ["vendor"], "role": "...", "text_only": false}]}

A component may list several roots under "paths". Nothing is ever written inside a component: all output goes to
`out`. Stdlib only, so the code-KB step can run under whatever interpreter has graphify installed."""
import json
import os
import subprocess
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".history", "graphify-out", ".terraform", "dist", "build", ".next", "Pods",
             "coverage", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".expo"}
DOC_SUFFIXES = {".md", ".mdx", ".rst", ".asc", ".adoc"}
# config/ops files: indexed as text (line windows), because devops answers live in them, not in prose
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".ini", ".tpl", ".patch", ".sh", ".example", ".env.example"}
CONFIG_NAMES = {"Dockerfile", "Makefile", "Procfile", ".env.example"}
CONFIG_SKIP = {"package-lock.json", "yarn.lock", "bun.lock", "pnpm-lock.yaml", "tsconfig.json", "composer.lock"}
SF_DATALESS = 0x40000000   # macOS: an iCloud file whose content is not on disk; reading it blocks on a download


class Component:
    def __init__(self, spec):
        self.name = spec["name"]
        self.roots = [Path(os.path.expanduser(p)).resolve() for p in spec.get("paths", [spec.get("path")]) if p]
        self.exclude = set(spec.get("exclude", []))
        self.role = spec.get("role", "")
        self.text_only = spec.get("text_only", False)
        self.skipped = []          # iCloud placeholders not read (see SF_DATALESS)

    def walk(self, suffixes=None):
        """Yield (root, path) for files under the roots, skipping SKIP_DIRS and this component's excludes."""
        for root in self.roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                rel = Path(dirpath).relative_to(root)
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and (not d.startswith(".") or d == ".github")
                                     and str(rel / d) not in self.exclude and d not in self.exclude)
                for f in sorted(filenames):
                    p = Path(dirpath) / f
                    if suffixes is None or p.suffix.lower() in suffixes or (suffixes == "config" and (
                            p.name in CONFIG_NAMES or (p.suffix.lower() in CONFIG_SUFFIXES and p.name not in CONFIG_SKIP
                                                       and p.stat().st_size < 200_000))):
                        if getattr(os.stat(p), "st_flags", 0) & SF_DATALESS:
                            if p not in self.skipped:
                                self.skipped.append(p)
                            continue
                        yield root, p

    def rel(self, root, path):
        """Stable display path: '<component>/<path inside root>' (plus the root's name if there are several)."""
        inner = path.relative_to(root).as_posix()
        return f"{self.name}/{root.name}/{inner}" if len(self.roots) > 1 else f"{self.name}/{inner}"


def load(path):
    spec = json.loads(Path(path).read_text())
    spec["out"] = Path(os.path.expanduser(spec.get("out", f"~/{spec['name']}-memory")))
    spec["components"] = [Component(c) for c in spec["components"]]
    return spec


def git_times(root, paths):
    """{path: last commit unix time} for files tracked in the git repo at `root`; files outside git fall back to mtime.
    One `git log` over the whole repo, not one call per file."""
    out = {}
    try:
        log = subprocess.run(["git", "-C", str(root), "log", "--format=%x01%ct", "--name-only"], capture_output=True,
                             text=True, timeout=120).stdout
        t = None
        for line in log.splitlines():
            if line.startswith("\x01"):
                t = int(line[1:])
            elif line and t is not None:
                out.setdefault((root / line).resolve(), t)       # first seen = most recent commit
    except (OSError, subprocess.SubprocessError):
        pass
    return {p: out.get(p.resolve(), int(p.stat().st_mtime)) for p in paths}
