"""Extractive handoff before a compaction (no model calls): Claude Code's PreCompact hook gives the transcript path;
this reads the JSONL transcript and writes a handoff note to the context repository, so after the compaction the
SessionStart `compact` hook brief (defrost brief) starts from the task, not from the lossy summary alone.

What is extracted, deterministically:
  goal        the first real user request of the session (+ the latest one, if different)
  state       the latest TodoWrite list: in progress / done counts and items
  next steps  pending TodoWrite items
  files       files written by Edit / Write / MultiEdit / NotebookEdit (latest last)
  decisions   questions the assistant asked last (lines ending in "?"), as open questions

Enabled per project with `defrost setup . --handoff-on-compact` (project .claude/settings.json only).
Inspired by Letta Code's reflection on compaction (src/agent/reflection-runs.ts, Apache-2.0), which uses an LLM
subagent; this version is extractive and free. An LLM summary step could replace `extract` (opt-in, not built)."""
from __future__ import annotations

import json
from pathlib import Path

HOOK_TAG = "defrost-ai:compact-handoff"
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def _is_request(text: str) -> bool:
    t = text.strip()
    return bool(t) and not t.startswith(("<", "[Request interrupted", "Caveat:")) and "tool_use_id" not in t


def extract(transcript: str | Path) -> dict:
    requests, todos, files, questions = [], None, [], []
    for line in Path(transcript).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        msg = e.get("message") or {}
        content = msg.get("content")
        if e.get("type") == "user" and not e.get("isMeta"):
            if isinstance(content, list) and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content):
                continue
            t = _text(content)
            if _is_request(t):
                requests.append(" ".join(t.split())[:600])
        elif e.get("type") == "assistant" and isinstance(content, list):
            asked = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use" and c.get("name") == "TodoWrite":
                    todos = (c.get("input") or {}).get("todos") or todos
                elif c.get("type") == "tool_use" and c.get("name") in EDIT_TOOLS:
                    f = (c.get("input") or {}).get("file_path") or (c.get("input") or {}).get("notebook_path")
                    if f:
                        files = [x for x in files if x != f] + [f]
                elif c.get("type") == "text":
                    asked += [s.strip() for s in c.get("text", "").splitlines() if s.strip().endswith("?")]
            if asked:
                questions = asked[-3:]
    goal = requests[0] if requests else ""
    if len(requests) > 1 and requests[-1] != goal:
        goal += f"\nLatest request: {requests[-1]}"
    todos = todos or []
    state = ""
    if todos:
        n = {s: sum(t.get("status") == s for t in todos) for s in ("completed", "in_progress", "pending")}
        state = (f"{n['completed']} done, {n['in_progress']} in progress, {n['pending']} pending. "
                 + " ".join(f"[{t.get('status')}] {t.get('content', '')}" for t in todos if t.get("status") != "completed"))
    return {"goal": goal, "state": state, "next_steps": [t.get("content", "") for t in todos if t.get("status") == "pending"],
            "files": files[-15:], "decisions": [f"open question: {q}" for q in questions]}


def run(payload: dict) -> Path | None:
    """PreCompact hook body: write the note; never raises into Claude Code (returns None on any problem)."""
    from defrost_ai import notes
    path = payload.get("transcript_path")
    domain = notes.domain_for(payload.get("cwd") or ".")
    if not path or not domain or not Path(path).exists():
        return None
    x = extract(path)
    if not x["goal"]:
        return None
    return notes.write_handoff(domain, x["goal"], x["state"], x["decisions"], x["next_steps"], x["files"],
                               source=f"auto, before {payload.get('trigger', 'auto')} compaction")


def install_hook(root: Path) -> str:
    import shutil
    f = root / ".claude/settings.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(f.read_text()) if f.exists() and f.read_text().strip() else {}
    hooks = settings.setdefault("hooks", {})
    exe = shutil.which("defrost") or "defrost"
    hooks["PreCompact"] = [h for h in hooks.get("PreCompact", []) if HOOK_TAG not in json.dumps(h)] + [
        {"hooks": [{"type": "command", "command": f"{exe} compact-handoff  # {HOOK_TAG}"}]}]
    f.write_text(json.dumps(settings, indent=2) + "\n")
    return str(f)


def remove_hook(root: Path) -> None:
    f = root / ".claude/settings.json"
    if not f.exists():
        return
    settings = json.loads(f.read_text() or "{}")
    pre = [h for h in settings.get("hooks", {}).get("PreCompact", []) if HOOK_TAG not in json.dumps(h)]
    if "hooks" in settings:
        settings["hooks"]["PreCompact"] = pre
        if not pre:
            settings["hooks"].pop("PreCompact")
        if not settings["hooks"]:
            settings.pop("hooks")
    f.write_text(json.dumps(settings, indent=2) + "\n")
