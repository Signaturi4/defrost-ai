"""Claude Code setup: slash commands + MCP server registration.

    defrost claude install [PROJECT_DIR] [--user]    # commands -> .claude/commands (or ~/.claude/commands)
                                                         # + `claude mcp add defrost -- defrost mcp`"""
from __future__ import annotations

import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

COMMANDS = ("defrost-setup.md", "ask.md", "handoff.md", "document-changes.md")
RETIRED = ("memory-search.md", "memory-update.md", "memory-init.md", "memory-domains.md")    # replaced in 1.2


def install(project: Path | None = None, user: bool = False, register_mcp: bool = True) -> list[str]:
    target = (Path.home() / ".claude/commands") if user else (Path(project or ".").resolve() / ".claude/commands")
    target.mkdir(parents=True, exist_ok=True)
    src = resources.files("defrost_ai.integrations.claude_commands")
    done = []
    for name in COMMANDS:
        (target / name).write_text(src.joinpath(name).read_text())
        done.append(str(target / name))
    for name in RETIRED:                                   # our own older commands only (they mention defrost/kev)
        old = target / name
        if old.exists() and any(w in old.read_text() for w in ("defrost", "kev-memory", "memory_search")):
            old.unlink()
            done.append(f"removed retired command {old}")
    if register_mcp:
        exe = shutil.which("defrost")
        cmd = [exe, "mcp"] if exe else [sys.executable, "-m", "defrost_ai.service.mcp_server"]
        scope = "user" if user else "project"
        claude = shutil.which("claude")
        if claude:
            cwd = None if user else Path(project or ".").resolve()
            subprocess.run([claude, "mcp", "remove", "defrost", "-s", scope], capture_output=True, text=True,
                           cwd=cwd)                        # re-register: the command path may have changed (rename)
            r = subprocess.run([claude, "mcp", "add", "defrost", "-s", scope, "--", *cmd], capture_output=True,
                               text=True, cwd=cwd)
            done.append(f"claude mcp add defrost -s {scope}: {'ok' if r.returncode == 0 else r.stderr.strip()[:200]}")
        else:
            done.append("claude CLI not found; register manually: claude mcp add defrost -- " + " ".join(cmd))
    return done
