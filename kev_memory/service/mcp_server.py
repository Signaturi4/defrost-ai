"""MCP server for defrost-ai (stdio): lets Claude Code, Claude Desktop, Cursor or any MCP client search and refresh
the memory. It holds no model itself: every call goes to the resident service (`kev-memory serve`), which it starts
on first use, so the models load once and are shared by all clients.

    claude mcp add defrost -- kev-memory mcp                   # Claude Code, all projects
    claude mcp add defrost -s project -- kev-memory mcp        # this project only (.mcp.json)

Tools: memory_search, memory_domains, memory_init, memory_update, memory_job, memory_rollback.
graphify users get the same search tools inside graphify's own MCP server via the patch in integrations/graphify/."""
from __future__ import annotations

import json
import os
from pathlib import Path

from kev_memory.service import client

HOME = Path(os.environ.get("KEV_MEMORY_HOME", "~/.kev-memory")).expanduser()
DEFAULT_DOMAINS = [d for d in os.environ.get("KEV_MEMORY_DOMAINS", "").split(",") if d]
MODES = ("fast", "rerank", "hybrid", "dense", "bm25", "all")


def _k(v):
    return "auto" if str(v) == "auto" else int(v)


def build_server():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("defrost", instructions=(
        "Project memory: documentation sections of the registered codebases and doc folders, each with the code it "
        "names. Use memory_search first for 'how do I / why does / what happens when' questions, and cite the "
        "returned path:Lstart-end. Use file reading or code-graph tools for structure and call paths."))

    @mcp.tool()
    def memory_search(query: str, domains: list[str] | None = None, mode: str = "fast", k: str = "auto") -> str:
        """Search the project memory. Returns the documentation sections that answer the question, cited as
        domain:path:Lstart-end, each followed by the code it names.
        mode: fast (default), rerank (most accurate, slower), dense (paraphrased how/why), bm25 (exact identifiers,
        flags, error strings), hybrid, all. k: number of sections (1-10) or "auto" (1-5 by retriever confidence).
        domains: restrict to these domains (default: KEV_MEMORY_DOMAINS or all registered)."""
        if mode not in MODES:
            return f"unknown mode {mode!r}; use one of {', '.join(MODES)}"
        res = client.search(query, domains or DEFAULT_DOMAINS or None, mode, _k(k), context=True)
        ctx = res.get("context")                     # the service renders the cited context pack
        return f"mode {res['mode']} -> {res['mode_used']}, {len(res['hits'])} sections\n\n{ctx or 'no results'}"

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
