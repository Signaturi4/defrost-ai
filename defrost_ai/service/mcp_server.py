"""MCP server for defrost-ai (stdio): four tools for Claude Code, Claude Desktop, Cursor or any MCP client.

    search     find the doc sections (with linked code) that answer a question; notes and decisions too
    docs_for   which doc sections describe the files you changed (or the current / staged change / a commit)
    remember   save a handoff note, or the user's decision on a doc/code conflict
    refresh    build or update the memory (also: status of every memory)

It holds no model itself: every call goes to the resident service (`defrost serve`), started on first use, so the
models load once and are shared by all clients.

    claude mcp add defrost -- defrost mcp                   # Claude Code, all projects
    claude mcp add defrost -s project -- defrost mcp        # this project only (.mcp.json)"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from defrost_ai import conflicts
from defrost_ai.service import client

DEFAULT_DOMAINS = [d for d in os.environ.get("DEFROST_DOMAINS", "").split(",") if d]


def _k(v):
    return "auto" if str(v) == "auto" else int(v)


def _here() -> str | None:
    from defrost_ai import notes
    return notes.domain_for(os.getcwd())


def _default_domains() -> list[str] | None:
    """The current project's memory plus its notes/decisions memory; else DEFROST_DOMAINS; else every memory."""
    if DEFAULT_DOMAINS:
        return DEFAULT_DOMAINS
    name = _here()
    if not name:
        return None
    from defrost_ai.context_repo import context_domain
    built = client.domains()
    return [d for d in (name, context_domain(name)) if built.get(d, {}).get("built")] or None


def build_server():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("defrost", instructions=(
        "Project memory: the documentation of this project, split into sections, each linked to the code and config "
        "files it names, plus saved handoff notes and the user's doc/code decisions. Call `search` first for 'how do "
        "I / why does / what happens when' questions and cite the returned path:Lstart-end. Docs can be out of date: "
        "each result starts with a 'doc trust' line set by the project owner (HIGH: answer from the docs; LOW: docs "
        "are hints, check the code); before stating how something behaves, read the files on a hit's 'verify in:' "
        "line, and always those behind a '!' line. After changing files, call `docs_for` and update those sections. "
        + conflicts.QUESTION))

    @mcp.tool()
    def search(question: str, mode: str | None = None, k: str = "auto", domains: list[str] | None = None) -> str:
        """Find the documentation sections that answer a question. Each hit is cited as domain:path:Lstart-end and is
        followed by the code it names, a 'verify in:' line (files to check the claim against), '!' lines when the doc
        looks stale or contradicts the code, and 'resolved:' when the user already decided a conflict there.
        mode: "accurate" (default: best results, ~1-2 s) or "fast" (no reranker, ~0.1 s, a little less accurate).
        k: number of sections (1-10) or "auto" (1-5, fewer when the top hit is clearly right).
        domains: limit to these memories (default: this project and its notes)."""
        try:
            res = client.search(question, domains or _default_domains(), mode, _k(k), context=True)
        except Exception as e:                                     # noqa: BLE001
            return f"search failed: {e}. If no memory exists yet, call refresh(path='.')."
        return f"{res['mode']} search: {len(res['hits'])} sections\n\n{res.get('context') or 'no results'}"

    @mcp.tool()
    def docs_for(files: list[str] | None = None, change: str = "working") -> str:
        """Which doc sections must be reviewed for a code change, so docs stay true.
        files: repo-relative paths you changed (e.g. ["config/deploy.yml"]) -> the sections that describe them.
        Without files, plans the docs for a whole change: change = "working" (uncommitted edits, default),
        "staged", or a commit sha. The plan also lists changed files no doc covers and new names (env vars, commands)
        that need a doc. Update those sections in the same change."""
        if files:
            hits = client.docs_for(files, _default_domains())
            return "\n".join(f"{h['file']}: {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}"
                             for h in hits) or "no doc section describes these files"
        from defrost_ai import docsync
        p = docsync.plan(os.getcwd(), staged=change == "staged", commit=None if change in ("working", "staged") else change)
        return docsync.render(p, 30) or "docs are up to date for this change"

    @mcp.tool()
    def remember(kind: str, goal: str = "", state: str = "", next_steps: list[str] | None = None,
                 decisions: list[str] | None = None, files: list[str] | None = None,
                 doc_path: str = "", decision: str = "", doc_says: str = "", code_does: str = "", code_ref: str = "",
                 doc_lines: list[int] | None = None, note: str = "") -> str:
        """Save something for later sessions (stored as a commit in the project's notes; searchable with `search`).
        kind="note": a handoff note before /clear or at the end of a session: goal (with acceptance criteria), state
          (done and verified), decisions (with reasons), next_steps, files. The next session starts from it.
        kind="decision": the USER's answer on a doc/code conflict, only after asking them (never decide yourself).
          decision: "code" (code is right, update the doc) | "doc" (doc is right, code is a bug) | "both" (not a
          conflict) | "open" (unsure). doc_path and doc_lines as cited by search; code_ref = file:line."""
        name = _here()
        if not name:
            return "no memory for this directory yet: call refresh(path='.') first"
        if kind == "note":
            if not goal:
                return "a note needs a goal"
            from defrost_ai import notes
            f = notes.write_handoff(name, goal, state, decisions or [], next_steps or [], files or [])
            try:
                notes.index(name, wait=False)
            except Exception:                                      # noqa: BLE001  (saved even if the service is down)
                pass
            return f"note saved: {f}. The user can run /clear; the next session starts from this note."
        if kind == "decision":
            try:
                row = conflicts.record(name, doc_path, decision, doc_says, code_does, code_ref, doc_lines, note)
            except ValueError as e:
                return str(e)
            return f"decision saved ({row['meaning']}); later search hits on {doc_path} show it."
        return 'kind must be "note" or "decision"'

    @mcp.tool()
    def refresh(path: str | None = None, wait: bool = True, status_only: bool = False) -> str:
        """Bring the memory up to date after docs or code changed (incremental: only edited sections are re-embedded).
        path: a repo or docs folder to index for the first time (stored outside the repo).
        status_only=true: list every memory with build time, size and whether it is stale; builds nothing."""
        if status_only:
            from defrost_ai.project_setup import status
            return json.dumps(status(), indent=1, default=str)
        if path:
            p = Path(path).expanduser().resolve()
            if not p.is_dir():
                return f"not a directory: {p}"
            from defrost_ai.project_setup import workspace_for
            from defrost_ai.library import register
            name = _here() or p.name.lower().replace(" ", "-")
            ws = workspace_for(p, name)
            register(name, ws, f"{p.name} (docs + code)")
        else:
            name = _here()
            if not name:
                return "no memory for this directory: call refresh(path='.') to create one"
        job = client.update(name, wait=False)
        return _wait(job["job"], name) if wait else f"refresh of {name} started (job {job['job']})"

    return mcp


def _wait(job_id: str, domain: str) -> str:
    import time
    while True:
        st = client.job(job_id)
        if st["state"] in ("done", "failed"):
            break
        time.sleep(2)
    if st["state"] == "failed":
        return f"build of {domain!r} failed:\n{st.get('error', '')[-1500:]}"
    m = st.get("manifest", {})
    return json.dumps({"domain": domain, "state": "done", "seconds": st.get("seconds"), "counts": m.get("counts"),
                       "changed": m.get("changed", {}).get("n")}, indent=1)


MISSING_MCP = ("defrost: the MCP server needs the `mcp` package, which is not installed in "
               f"{sys.prefix}.\nMCP clients only report this as 'connection closed'. Reinstall defrost-ai:\n"
               "  curl -fsSL https://raw.githubusercontent.com/Signaturi4/defrost-ai/main/install.sh | sh\n"
               "  or: uv tool install --reinstall \"defrost-ai[code,mac] @ <source>\"\n"
               "  or: pip install \"mcp>=1.2,<2\" (into this environment)")


def mcp_available() -> bool:
    """True when `mcp` is importable. It is a core dependency, but a hand-built or partial environment can still lack it,
    and the server cannot start without it."""
    return importlib.util.find_spec("mcp") is not None


def main():
    if not mcp_available():
        print(MISSING_MCP, file=sys.stderr)              # stderr reaches the client's MCP log, not the protocol
        sys.exit(1)
    build_server().run()


if __name__ == "__main__":
    main()
