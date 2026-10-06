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
TIMEOUT = 4.0                                            # seconds for a fast search (it takes ~0.05 s warm)
RERANK_TIMEOUT = 5.0                                     # seconds for a reranked search (~1-2 s warm); then fast
GUIDE = ("Report every disagreement you see as a conflict, with both versions and their path:line: between two "
         "sections, between a section and the code, and a section's own note that the code differs or that "
         "something is not built yet. Say which version the code follows. Answer a plain yes or no only when the "
         "evidence supports all of it; otherwise say what holds and what does not.")


def _worth_searching(prompt: str) -> bool:
    p = prompt.strip()
    return bool(p) and not p.startswith("/") and len(p.split()) >= MIN_WORDS


def _start_service_in_background() -> None:
    subprocess.Popen([sys.executable, "-c", "from defrost_ai.service.client import ensure_service; ensure_service()"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def context_for(payload: dict) -> str | None:
    """The text to add to Claude's context for this prompt, or None."""
    prompt = payload.get("prompt") or ""
    if not _worth_searching(prompt):
        return None
    from defrost_ai import notes, settings
    from defrost_ai.context_repo import context_domain
    from defrost_ai.service import client
    name = notes.domain_for(payload.get("cwd") or os.getcwd())
    if not name:
        return None
    if not client.alive():                               # never wait for a cold start inside the user's prompt
        _start_service_in_background()
        return None
    built = client._call("GET", "/domains", timeout=TIMEOUT)
    domains = [d for d in (name, context_domain(name)) if built.get(d, {}).get("built")]
    if not domains:
        return None
    budget = int(settings.get("prompt_context.budget_tokens"))
    query = {"query": prompt.strip()[:MAX_QUERY], "domains": domains, "k": int(settings.get("prompt_context.k")),
             "context": True, "budget_tokens": budget}
    res = None
    if settings.get("prompt_context.mode") == "accurate":   # reranked: puts both sides of a conflict in the top k
        try:
            res = client._call("POST", "/search", query | {"mode": "accurate"}, timeout=RERANK_TIMEOUT)
        except Exception:                                    # a cold reranker must not cost the prompt its context
            res = None
    if res is None:
        res = client._call("POST", "/search", query | {"mode": "fast"}, timeout=TIMEOUT)
    hits = res.get("hits") or []
    best = max((h.get("cosine") or 0 for h in hits), default=0)
    if best < settings.get("prompt_context.min_cosine") or not res.get("context"):
        return None
    return (f"Project memory ({name}) was searched automatically for this prompt (best match {best:.2f}). If these "
            "sections answer it, answer from them and cite path:Lstart-end without searching again; call the defrost "
            f"`search` tool only for what they do not cover. {GUIDE}\n\n" + res["context"][:budget * 4 + 2000])


def hook(payload: dict) -> dict | None:
    text = context_for(payload)
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
