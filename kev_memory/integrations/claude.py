"""Claude Code setup: slash commands + MCP server registration.

    kev-memory claude install [PROJECT_DIR] [--user]    # commands -> .claude/commands (or ~/.claude/commands)
                                                         # + `claude mcp add defrost -- kev-memory mcp`"""
from __future__ import annotations

import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

COMMANDS = ("defrost-setup.md", "memory-search.md", "memory-update.md", "memory-init.md", "memory-domains.md")


def install(project: Path | None = None, user: bool = False, register_mcp: bool = True) -> list[str]:
    target = (Path.home() / ".claude/commands") if user else (Path(project or ".").resolve() / ".claude/commands")
    target.mkdir(parents=True, exist_ok=True)
    src = resources.files("kev_memory.integrations.claude_commands")
    done = []
    for name in COMMANDS:
        (target / name).write_text(src.joinpath(name).read_text())
        done.append(str(target / name))
    if register_mcp:
        exe = shutil.which("kev-memory")
        cmd = [exe, "mcp"] if exe else [sys.executable, "-m", "kev_memory.service.mcp_server"]
        scope = "user" if user else "project"
        claude = shutil.which("claude")
        if claude:
            r = subprocess.run([claude, "mcp", "add", "defrost", "-s", scope, "--", *cmd], capture_output=True,
                               text=True, cwd=None if user else Path(project or ".").resolve())
            done.append(f"claude mcp add defrost -s {scope}: {'ok' if r.returncode == 0 else r.stderr.strip()[:200]}")
        else:
            done.append("claude CLI not found; register manually: claude mcp add defrost -- " + " ".join(cmd))
    return done
