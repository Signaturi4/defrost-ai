"""Working memory for Claude Code sessions: handoff notes instead of long conversation history.

A session writes a handoff note (goal, state, decisions, next steps, files) before `/clear`. The note is a small
Markdown page in a per-project notes folder that is indexed as its own memory domain, `<domain>-notes`, so
`memory_search` finds old decisions later. A SessionStart hook (matchers `clear` and `compact`) prints only the
latest note's brief (goal + state + next steps + files), so the new context holds the task, not the transcript.

    kev-memory handoff --goal "..." --state "..." --next "..." --file path   # write a note (also MCP memory_handoff)
    kev-memory brief                                                         # what the hook prints ("" if no notes)

Notes live in ~/.kev-memory/<domain>-notes/notes/, outside the repository: they hold transient session state
(half-made decisions, failed attempts) that should not land in git or in the project's docs domain."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

BRIEF_WORDS = 350                                                # hard cap on what the hook injects
HOOK_TAG = "defrost-ai:handoff"


def home() -> Path:
    return Path(os.environ.get("KEV_MEMORY_HOME", "~/.kev-memory")).expanduser()


def notes_domain(domain: str) -> str:
    return f"{domain}-notes"


def notes_dir(domain: str) -> Path:
    return home() / notes_domain(domain) / "notes"


def domain_for(path: str | Path = ".") -> str | None:
    """The registered domain whose component contains `path` (the deepest one), else None."""
    p = Path(path).expanduser().resolve()
    best, depth = None, -1
    for ws in home().glob("*.workspace.json"):
        try:
            data = json.loads(ws.read_text())
        except (OSError, ValueError):
            continue
        if data["name"].endswith("-notes"):
            continue
        for c in data.get("components", []):
            root = Path(c["path"]).expanduser().resolve()
            if (p == root or root in p.parents) and len(root.parts) > depth:
                best, depth = data["name"], len(root.parts)
    return best


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "note"


def _bullets(items) -> str:
    items = [items] if isinstance(items, str) else list(items or [])
    return "\n".join(f"- {i.strip()}" for i in items if i and i.strip()) or "- (none)"


def write_handoff(domain: str, goal: str, state: str = "", decisions=(), next_steps=(), files=(),
                  when: float | None = None) -> Path:
    """One note = one page; every section names the goal, so each one makes sense alone in search results."""
    when = time.time() if when is None else when
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(when))
    short = goal.strip().splitlines()[0][:80]
    d = notes_dir(domain)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(when))}-{_slug(short)}.md"
    f.write_text(
        f"# Handoff {stamp}: {short}\n\n"
        f"## Goal ({short})\n{goal.strip()}\n\n"
        f"## State of the work ({short})\n{state.strip() or '(not given)'}\n\n"
        f"## Decisions ({short})\n{_bullets(decisions)}\n\n"
        f"## Next steps ({short})\n{_bullets(next_steps)}\n\n"
        f"## Files ({short})\n{_bullets(f'`{x}`' for x in files) if files else '- (none)'}\n")
    return f


def latest(domain: str) -> Path | None:
    d = notes_dir(domain)
    notes = sorted(d.glob("*.md")) if d.exists() else []
    return notes[-1] if notes else None


def _section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}\b.*?$\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    return m.group(1).strip() if m else ""


def brief(domain: str | None, max_words: int = BRIEF_WORDS) -> str:
    """Goal + state + next steps + files of the newest note, capped at `max_words` words."""
    if not domain:
        return ""
    f = latest(domain)
    if f is None:
        return ""
    text = f.read_text()
    parts = [("Goal", _section(text, "Goal")), ("State", _section(text, "State of the work")),
             ("Next steps", _section(text, "Next steps")), ("Files", _section(text, "Files"))]
    body = "\n".join(f"{k}: {v}" if "\n" not in v else f"{k}:\n{v}" for k, v in parts if v and v != "- (none)")
    words = body.split(" ")
    if len(words) > max_words:
        body = " ".join(words[:max_words]) + " …"
    return (f"[defrost-ai handoff, {f.stem[:15]}] Continue from this note, not from memory of the old conversation.\n"
            f"{body}\n"
            f"Older decisions and notes: memory_search(query, domains=[\"{notes_domain(domain)}\"]). "
            f"Before ending or before /clear, write a new note with memory_handoff.")


def register_notes(domain: str) -> Path:
    """Workspace + registry entry for the notes domain (docs only), so the normal builder indexes it."""
    from kev_memory.library import register
    home().mkdir(parents=True, exist_ok=True)
    ws = home() / f"{notes_domain(domain)}.workspace.json"
    if not ws.exists():
        ws.write_text(json.dumps({"name": notes_domain(domain), "out": str(home() / notes_domain(domain) / "memory"),
                                  "components": [{"name": notes_domain(domain), "path": str(notes_dir(domain))}]},
                                 indent=1))
    register(notes_domain(domain), ws, f"session handoff notes for {domain}")
    return ws


def index(domain: str, wait: bool = False) -> dict:
    """Incremental rebuild of the notes domain through the resident service (only the new note is embedded)."""
    from kev_memory.service import client
    register_notes(domain)
    client.ensure_service()
    return client.update(notes_domain(domain), wait=wait)


# ---- Claude Code wiring (project level only) -------------------------------------------------------------------
def install_hook(root: Path) -> str:
    """Project .claude/settings.json: after /clear or a compaction, print the latest brief into the new context."""
    import shutil
    f = root / ".claude/settings.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(f.read_text()) if f.exists() and f.read_text().strip() else {}
    hooks = settings.setdefault("hooks", {})
    starts = [h for h in hooks.get("SessionStart", []) if HOOK_TAG not in json.dumps(h)]
    exe = shutil.which("kev-memory") or "kev-memory"
    starts.append({"matcher": "clear|compact",
                   "hooks": [{"type": "command", "command": f"{exe} brief  # {HOOK_TAG}"}]})
    hooks["SessionStart"] = starts
    f.write_text(json.dumps(settings, indent=2) + "\n")
    return str(f)


def remove_hook(root: Path) -> None:
    f = root / ".claude/settings.json"
    if not f.exists():
        return
    settings = json.loads(f.read_text() or "{}")
    starts = [h for h in settings.get("hooks", {}).get("SessionStart", []) if HOOK_TAG not in json.dumps(h)]
    if "hooks" in settings:
        settings["hooks"]["SessionStart"] = starts
        if not starts:
            settings["hooks"].pop("SessionStart")
        if not settings["hooks"]:
            settings.pop("hooks")
    f.write_text(json.dumps(settings, indent=2) + "\n")


def git_root(path: Path) -> Path:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else Path(path).resolve()
