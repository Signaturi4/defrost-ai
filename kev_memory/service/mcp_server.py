"""MCP server for defrost-ai (stdio): lets Claude Code, Claude Desktop, Cursor or any MCP client search and refresh
the memory. It holds no model itself: every call goes to the resident service (`kev-memory serve`), which it starts
on first use, so the models load once and are shared by all clients.

    claude mcp add defrost -- kev-memory mcp                   # Claude Code, all projects
    claude mcp add defrost -s project -- kev-memory mcp        # this project only (.mcp.json)

Tools: memory_search, memory_docs_for, memory_domains, memory_init, memory_update, memory_job, memory_rollback,
memory_handoff, memory_brief (session working memory, see kev_memory/notes.py), memory_resolve_conflict,
memory_conflicts (doc/code conflicts are decided by the user, see kev_memory/conflicts.py), memory_context_log
(history of the git-backed working memory, see kev_memory/context_repo.py).
graphify users get the same search tools inside graphify's own MCP server via the patch in integrations/graphify/."""
from __future__ import annotations

import json
import os
from pathlib import Path

from kev_memory import conflicts
from kev_memory.service import client

HOME = Path(os.environ.get("KEV_MEMORY_HOME", "~/.kev-memory")).expanduser()
DEFAULT_DOMAINS = [d for d in os.environ.get("KEV_MEMORY_DOMAINS", "").split(",") if d]
MODES = ("fast", "rerank", "hybrid", "dense", "bm25", "all")


def _k(v):
    return "auto" if str(v) == "auto" else int(v)


def build_server():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("defrost", instructions=(
        "Project memory: documentation sections of the registered codebases and doc folders, each with the code and "
        "config files it names. Use memory_search first for 'how do I / why does / what happens when' questions, and "
        "cite the returned path:Lstart-end. Docs can be out of date: before stating how something behaves, read the "
        "files on each hit's 'verify in:' line (config files first) and always the ones marked 'doc may be stale'. "
        "Each result starts with a 'doc trust' line set by the project owner (HIGH: answer from the docs; LOW: docs "
        "are hints, answer from the code): follow it. " + conflicts.QUESTION))

    @mcp.tool()
    def memory_search(query: str, domains: list[str] | None = None, mode: str = "fast", k: str = "auto") -> str:
        """Search the project memory. Returns the documentation sections that answer the question, cited as
        domain:path:Lstart-end, each followed by the code it names, a 'verify in:' line with the code/config files to
        check the claim against, and a '! doc may be stale' line when such a file changed after the doc.
        mode: fast (default), rerank (most accurate, slower), dense (paraphrased how/why), bm25 (exact identifiers,
        flags, error strings), hybrid, all. k: number of sections (1-10) or "auto" (1-5 by retriever confidence).
        domains: restrict to these domains (default: KEV_MEMORY_DOMAINS or all registered)."""
        if mode not in MODES:
            return f"unknown mode {mode!r}; use one of {', '.join(MODES)}"
        res = client.search(query, domains or DEFAULT_DOMAINS or None, mode, _k(k), context=True)
        ctx = res.get("context")                     # the service renders the cited context pack
        return f"mode {res['mode']} -> {res['mode_used']}, {len(res['hits'])} sections\n\n{ctx or 'no results'}"

    @mcp.tool()
    def memory_docs_for(paths: list[str], domains: list[str] | None = None) -> str:
        """Doc sections that describe the given code or config files (repo-relative paths, e.g. "config/deploy.yml").
        Call it after changing those files: these are the docs to review and update in the same change."""
        hits = client.docs_for(paths, domains or DEFAULT_DOMAINS or None)
        if not hits:
            return "no doc section links to these files"
        return "\n".join(f"{h['file']}: {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}"
                         for h in hits)

    @mcp.tool()
    def memory_resolve_conflict(domain: str, doc_path: str, decision: str, doc_says: str = "", code_does: str = "",
                                code_ref: str = "", doc_lines: list[int] | None = None, note: str = "") -> str:
        """Record the USER's decision on a doc/code conflict, after asking them (never decide it yourself).
        decision: "code" (code is right, update the doc) | "doc" (doc is right, the code is a bug) | "both" (not a
        conflict) | "open" (unsure: mark as open question). doc_path/doc_lines as cited in memory_search
        (domain:path:Lstart-end -> doc_path=path); code_ref = file:line. Later search hits show the decision."""
        try:
            return json.dumps(conflicts.record(domain, doc_path, decision, doc_says, code_does, code_ref, doc_lines,
                                               note), indent=1)
        except ValueError as e:
            return str(e)

    @mcp.tool()
    def memory_conflicts(domain: str) -> str:
        """Doc/code conflicts the user has decided for this domain (latest last)."""
        return json.dumps(conflicts.load(domain), indent=1)

    @mcp.tool()
    def memory_domains() -> str:
        """List the registered memory domains (repos / doc folders) with their sources, build time and counts."""
        return json.dumps(client.domains(), indent=1)

    @mcp.tool()
    def memory_init(path: str, domain: str | None = None, description: str = "") -> str:
        """Create a memory for a repo or docs folder and build it (docs + code). Stored in ~/.kev-memory/<domain>,
        never inside the repo. Returns when the build finishes (minutes for a large repo)."""
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return f"not a directory: {p}"
        name = domain or p.name.lower().replace(" ", "-")
        HOME.mkdir(parents=True, exist_ok=True)
        ws = HOME / f"{name}.workspace.json"
        ws.write_text(json.dumps({"name": name, "out": str(HOME / name),
                                  "components": [{"name": name, "path": str(p)}]}, indent=1))
        client.ensure_service()
        job = client._call("POST", "/update", {"domain": name, "workspace": str(ws), "description": description})
        return _wait(job["job"], name)

    @mcp.tool()
    def memory_update(domain: str, wait: bool = True) -> str:
        """Rebuild a domain after docs or code changed. Incremental: only new or edited sections are re-embedded;
        the previous build is kept for memory_rollback."""
        job = client.update(domain, wait=False)
        return _wait(job["job"], domain) if wait else json.dumps({"job": job["job"], "state": "queued"})

    @mcp.tool()
    def memory_job(job_id: str) -> str:
        """State of a build job started with memory_update(wait=false)."""
        return json.dumps({k: v for k, v in client.job(job_id).items() if k != "log"}, indent=1, default=str)

    @mcp.tool()
    def memory_rollback(domain: str) -> str:
        """Restore the previous build of a domain."""
        return json.dumps(client.rollback(domain))

    @mcp.tool()
    def memory_handoff(goal: str, state: str = "", decisions: list[str] | None = None,
                       next_steps: list[str] | None = None, files: list[str] | None = None,
                       domain: str | None = None) -> str:
        """Save a handoff note for this project so the work can continue after /clear with a small context:
        goal (one paragraph, with acceptance criteria), state (what is done and verified), decisions (with the
        reason), next_steps (concrete), files (paths that matter). The note is committed to the project's context
        repository and indexed as the '<domain>-context' memory; after /clear the SessionStart hook shows its brief. Call it before /clear or when the user ends a
        session. domain: default = the memory domain of the current directory."""
        from kev_memory import notes
        name = domain or notes.domain_for(os.getcwd())
        if not name:
            return "no memory domain for this directory; run /defrost-setup first or pass domain"
        f = notes.write_handoff(name, goal, state, decisions or [], next_steps or [], files or [])
        try:
            job = notes.index(name, wait=False)
            indexed = f"indexing job {job.get('job')}"
        except Exception as e:                                # noqa: BLE001  (note is saved even if the service is down)
            indexed = f"not indexed yet ({e}); run memory_update('{notes.notes_domain(name)}')"
        return f"handoff saved: {f}\n{indexed}\nThe user can now run /clear; the new session starts from this note."

    @mcp.tool()
    def memory_docs_plan(commit: str | None = None, staged: bool = False) -> str:
        """Which doc sections describe the code changed in this repo (working tree vs HEAD by default, or the staged
        changes, or one commit), which changed files have no doc yet, and which docs were already edited. Model-free.
        Use it before writing docs for a change (/document-changes)."""
        from kev_memory import docsync
        p = docsync.plan(os.getcwd(), staged=staged, commit=commit)
        return docsync.render(p, 30) or "docs up to date for this change"

    @mcp.tool()
    def memory_brief(domain: str | None = None) -> str:
        """The latest handoff note of this project as a short brief (goal, state, next steps, files), plus the map of
        its context repository (open a folder's MEMORY.md for more; search domain '<domain>-context')."""
        from kev_memory import notes
        return notes.brief(domain or notes.domain_for(os.getcwd())) or "no handoff notes yet"

    @mcp.tool()
    def memory_context_log(domain: str | None = None, n: int = 15) -> str:
        """Audit trail of the project's working memory: the latest commits of its context repository (handoff notes,
        the user's doc/code decisions, maintenance), newest first."""
        from kev_memory import context_repo, notes
        name = domain or notes.domain_for(os.getcwd())
        rows = context_repo.log(name, n) if name else []
        return "\n".join(f"{r['sha']} {r['date']} {r['author']}: {r['subject']}" for r in rows) or "no context history"

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


def main():
    build_server().run()


if __name__ == "__main__":
    main()
