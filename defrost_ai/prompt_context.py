"""Claude Code UserPromptSubmit hook: search the project memory for each question before Claude starts answering.

Why: in an MCP-only setup every answer costs extra model turns (find the deferred `search` tool, call it, then
answer), measured at 18-30 s for "who is artem" while the search itself takes ~0.05 s. Putting the best sections into
the prompt's context lets Claude answer in one turn (~4 s with Sonnet), and it can still call `search` for more.

Only adds context when it is likely to help: slash commands and 1-2 word prompts ("ok", "continue") are skipped, and
so is anything whose best section is less similar than `prompt_context.min_cosine` (coding requests, chit-chat).
Never blocks or slows a prompt: when the service is not running it is started in the background and this prompt
gets no context; every error is swallowed by `defrost hook`."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HOOK_TAG = "defrost-ai:prompt-context"
EVENT = "UserPromptSubmit"
MIN_WORDS = 3
MAX_QUERY = 500                                          # characters of the prompt used as the search query
MAX_CHARS = 8000                                         # characters of context added to the prompt
K = 3
TIMEOUT = 4.0                                            # seconds for the search call (it takes ~0.05 s warm)


def _worth_searching(prompt: str) -> bool:
    p = prompt.strip()
    return bool(p) and not p.startswith("/") and len(p.split()) >= MIN_WORDS


def _start_service_in_background() -> None:
    subprocess.Popen([sys.executable, "-c", "from defrost_ai.service.client import ensure_service; ensure_service()"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def context_for(payload: dict, info: dict | None = None) -> str | None:
    """The text to add to Claude's context for this prompt, or None. `info` receives why (for monitoring)."""
    info = {} if info is None else info
    prompt = payload.get("prompt") or ""
    if not _worth_searching(prompt):
        info["reason"] = "command or short prompt"
        return None
    from defrost_ai import notes, settings
    from defrost_ai.context_repo import context_domain
    from defrost_ai.service import client
    name = notes.domain_for(payload.get("cwd") or os.getcwd())
    if not name:
        info["reason"] = "no memory for this folder"
        return None
    if not client.alive():                               # never wait for a cold start inside the user's prompt
        _start_service_in_background()
        info["reason"] = "service not running (started in background)"
        return None
    built = client._call("GET", "/domains", timeout=TIMEOUT)
    domains = [d for d in (name, context_domain(name)) if built.get(d, {}).get("built")]
    if not domains:
        info["reason"] = "memory not built"
        return None
    res = client._call("POST", "/search", {"query": prompt.strip()[:MAX_QUERY], "domains": domains, "mode": "fast",
                                           "k": K, "context": True}, timeout=TIMEOUT)
    hits = res.get("hits") or []
    best = max((h.get("cosine") or 0 for h in hits), default=0)
    info.update(best=round(best, 3), hits=len(hits), domain=name)
    if best < settings.get("prompt_context.min_cosine") or not res.get("context"):
        info["reason"] = "below relevance threshold"
        return None
    return (f"Project memory ({name}) was searched automatically for this prompt (best match {best:.2f}). If these "
            "sections answer it, answer from them and cite path:Lstart-end without searching again; call the defrost "
            "`search` tool only for what they do not cover.\n\n" + res["context"][:MAX_CHARS])


def hook(payload: dict) -> dict | None:
    import time
    from defrost_ai import monitor_event
    t0, info, text, err = time.time(), {}, None, None
    try:
        text = context_for(payload, info)
    except Exception as e:                               # noqa: BLE001  (logged, then re-raised to `defrost hook`)
        err = e
    monitor_event("memory_inject", payload.get("session_id"), payload.get("cwd"),
                  decision="injected" if text else ("error" if err else "skipped"),
                  reason=f"{type(err).__name__}: {err}" if err else info.get("reason"),
                  ms=round((time.time() - t0) * 1000), chars=len(text or ""),
                  **{k: v for k, v in info.items() if k != "reason"})
    if err:
        raise err
    return {"hookSpecificOutput": {"hookEventName": EVENT, "additionalContext": text}} if text else None


# ---- Claude Code wiring (project .claude/settings.json) ------------------------------------------------------------
def install_hook(root: Path) -> str:
    from defrost_ai.notes import claude_hook_command
    f = root / ".claude/settings.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(f.read_text()) if f.exists() and f.read_text().strip() else {}
    hooks = settings.setdefault("hooks", {})
    entries = [h for h in hooks.get(EVENT, []) if HOOK_TAG not in json.dumps(h)]
    entries.append({"hooks": [{"type": "command", "command": claude_hook_command("hook prompt", HOOK_TAG),
                               "timeout": 10}]})
    hooks[EVENT] = entries
    f.write_text(json.dumps(settings, indent=2) + "\n")
    return str(f)


def remove_hook(root: Path) -> None:
    f = root / ".claude/settings.json"
    if not f.exists():
        return
    settings = json.loads(f.read_text() or "{}")
    hooks = settings.get("hooks", {})
    if EVENT not in hooks:
        return
    entries = [h for h in hooks[EVENT] if HOOK_TAG not in json.dumps(h)]
    if entries:
        hooks[EVENT] = entries
    else:
        hooks.pop(EVENT)
        if not hooks:
            settings.pop("hooks")
    f.write_text(json.dumps(settings, indent=2) + "\n")
