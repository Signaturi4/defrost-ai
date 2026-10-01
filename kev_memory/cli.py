"""kev-memory command line.

  kev-memory build WORKSPACE.json [--domain NAME] [--description TEXT]   build or update (incremental) a memory
  kev-memory update DOMAIN                                               rebuild a registered domain
  kev-memory rollback DOMAIN                                             restore the previous build
  kev-memory setup [PATH] [--on-main-merge] [--every-hours N] [--claude-hook] [--doc-rules] [--claude] [--doc-trust high|low]
                                                                         one-shot project setup + refresh triggers
  kev-memory refresh DOMAIN [--if-changed]                               what the triggers run
  kev-memory status [DOMAIN]                                             staleness + installed triggers
  kev-memory download-weights                                            fetch + verify the model weights (auto on first use)
  kev-memory mcp                                                         MCP server (stdio) for Claude Code / Desktop
  kev-memory claude install [DIR] [--user]                               slash commands + `claude mcp add defrost`
  kev-memory domains                                                     list registered domains
  kev-memory search "question" [--domain D ...] [--mode fast] [-k 5] [--json]
  kev-memory docs-for config/deploy.yml [...]                           doc sections to review after editing these files
  kev-memory serve [--port 8765]                                         resident HTTP service
  kev-memory benchmark --suite FILE.jsonl --memory DIR [--split dev]
  kev-memory verify-weights                                              check model files against MANIFEST.json"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kev-memory", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("workspace"); b.add_argument("--domain"); b.add_argument("--description", default="")
    u = sub.add_parser("update"); u.add_argument("domain")
    r = sub.add_parser("rollback"); r.add_argument("domain")
    sub.add_parser("domains")
    df = sub.add_parser("docs-for", help="doc sections that describe these code/config files (review after editing)")
    df.add_argument("paths", nargs="+"); df.add_argument("--domain", action="append")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--domain", action="append")
    s.add_argument("--mode", default="fast"); s.add_argument("-k", type=lambda v: v if v == "auto" else int(v), default=5,
                   help='number of sections, or "auto" (adaptive, 1-5)'); s.add_argument("--json", action="store_true")
    s.add_argument("--merge", default="rerank", choices=["rerank", "rrf"])
    s.add_argument("--local", action="store_true", help="load the models in this process instead of using the service")
    v = sub.add_parser("serve"); v.add_argument("--port", type=int, default=8765); v.add_argument("--host", default="127.0.0.1")
    e = sub.add_parser("benchmark"); e.add_argument("--suite", required=True); e.add_argument("--memory", required=True)
    e.add_argument("--split", default="dev"); e.add_argument("--out"); e.add_argument("--rankings")
    e.add_argument("--only", help="keep suite rows whose \"suite\" field matches (multi-memory suites)")
    sub.add_parser("verify-weights")
    st = sub.add_parser("setup", help="one-shot project setup + refresh triggers")
    st.add_argument("path", nargs="?", default="."); st.add_argument("--domain")
    st.add_argument("--build", choices=["now", "background", "skip"], default="now")
    st.add_argument("--on-main-merge", action="store_true", help="git hooks: refresh when changes land on main/master")
    st.add_argument("--every-hours", type=float, help="refresh every N hours (launchd on macOS, cron on Linux)")
    st.add_argument("--claude-hook", action="store_true", help="Claude Code SessionStart hook: refresh when stale")
    st.add_argument("--doc-rules", action="store_true", help="add the doc-writing rules block to CLAUDE.md")
    st.add_argument("--claude", action="store_true", help="install slash commands + register the MCP server")
    st.add_argument("--remove-triggers", action="store_true")
    st.add_argument("--doc-trust", choices=["high", "low"],
                    help="high: docs are reliable, answer from them; low (default): docs are hints, always check code")
    rf = sub.add_parser("refresh"); rf.add_argument("domain"); rf.add_argument("--if-changed", action="store_true")
    ss = sub.add_parser("status"); ss.add_argument("domain", nargs="?")
    sub.add_parser("download-weights")
    sub.add_parser("mcp", help="MCP server over stdio (claude mcp add defrost -- kev-memory mcp)")
    c = sub.add_parser("claude", help="Claude Code setup: slash commands + MCP registration")
    c.add_argument("action", choices=["install"]); c.add_argument("project", nargs="?", default=".")
    c.add_argument("--user", action="store_true", help="install for all projects (~/.claude/commands, user scope)")
    c.add_argument("--no-mcp", action="store_true", help="only copy the slash commands")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from kev_memory.builder import build
        from kev_memory.config import Workspace
        from kev_memory.library import register
        build(a.workspace)
        if a.domain:
            register(a.domain, a.workspace, a.description)
            print(f"registered domain {a.domain!r}")
    elif a.cmd == "update":
        from kev_memory.builder import build
        from kev_memory.library import read_registry
        build(read_registry()["domains"][a.domain]["workspace"])
    elif a.cmd == "rollback":
        from kev_memory.builder import rollback
        from kev_memory.library import read_registry
        print("rolled back" if rollback(read_registry()["domains"][a.domain]["workspace"]) else "no previous build")
    elif a.cmd == "domains":
        from kev_memory.library import Library
        print(json.dumps(Library().domains(), indent=1))
    elif a.cmd == "docs-for":
        from kev_memory.library import Library
        hits = Library().docs_for(a.paths, a.domain)
        for h in hits:
            print(f"{h['file']}: {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}")
        if not hits:
            print("no doc section links to these files")
    elif a.cmd == "search":
        from kev_memory.memory import Memory
        if a.local:                                      # load the models in this process (slow: every call)
            from kev_memory.library import Library
            res = Library().search(a.query, a.domain, a.mode, a.k, a.merge)
        else:                                            # resident service: models stay loaded between calls
            from kev_memory.service import client
            client.ensure_service()
            res = client.search(a.query, a.domain, a.mode, a.k, context=False, merge=a.merge)
        print(json.dumps(res, indent=1) if a.json else f"mode {res['mode']} -> {res['mode_used']}\n\n" + Memory.context(res))
    elif a.cmd == "serve":
        from kev_memory.service.http_server import serve
        serve(a.port, a.host)
    elif a.cmd == "benchmark":
        from kev_memory.evaluation.benchmark import print_report, run
        res = run(a.suite, a.memory, a.split, save_rankings=a.rankings, only=a.only)
        print_report(res)
        if a.out:
            open(a.out, "w").write(json.dumps(res, indent=1))
    elif a.cmd == "setup":
        from kev_memory.project_setup import setup
        res = setup(a.path, a.domain, a.build, a.on_main_merge, a.every_hours, a.claude_hook, a.doc_rules, a.claude,
                    a.remove_triggers, doc_trust=a.doc_trust)
        print(json.dumps(res, indent=1, default=str))
    elif a.cmd == "refresh":
        from kev_memory.project_setup import refresh
        res = refresh(a.domain, a.if_changed, log=lambda m: print(time.strftime("%F %T"), m, flush=True))
        return 0 if res.get("job") != "failed" else 1
    elif a.cmd == "status":
        from kev_memory.project_setup import status
        print(json.dumps(status(a.domain), indent=1))
    elif a.cmd == "download-weights":
        from kev_memory.models.weights import download_weights, models_dir
        print(f"weights ready: {models_dir()}") if _weights_ok() else download_weights()
    elif a.cmd == "mcp":
        from kev_memory.service.mcp_server import main as mcp_main
        mcp_main()
    elif a.cmd == "claude":
        from kev_memory.integrations.claude import install
        for line in install(Path(a.project), user=a.user, register_mcp=not a.no_mcp):
            print(line)
    elif a.cmd == "verify-weights":
        from kev_memory.models.weights import models_dir, verify
        res = verify()
        print(json.dumps({"dir": str(models_dir()), **res}, indent=1))
        return 0 if res["ok"] else 1
    return 0


def _weights_ok() -> bool:
    try:
        from kev_memory.models.weights import CACHE, WEIGHTS_VERSION, models_dir, verify
        d = models_dir(download=False)
        return verify(d)["ok"] and (d != CACHE / "models" or verify(d)["version"] == WEIGHTS_VERSION)
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
