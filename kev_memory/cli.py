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
  kev-memory verify-weights                                              check model files against MANIFEST.json
  kev-memory context init|check|log|brief|defrag|branches|merge|remote   the project's git-backed working memory"""
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
    st.add_argument("--handoff", action="store_true",
                    help="Claude Code: after /clear or compaction, start from the latest handoff note (project hook)")
    st.add_argument("--docs-sync", action="store_true",
                    help="Claude Code: document changes after editing and before git commit; post-commit doc tasks")
    st.add_argument("--docs-auto", action="store_true",
                    help="also run claude -p /document-changes after every commit made outside Claude (costs tokens)")
    st.add_argument("--docs-budget", type=float, default=0.5, help="USD cap per automatic run (default 0.5)")
    st.add_argument("--docs-auto-merge", action="store_true",
                    help="fast-forward the docs-auto branch into the checkout (default: keep defrost/docs/<sha> for review)")
    st.add_argument("--handoff-on-compact", action="store_true",
                    help="Claude Code PreCompact hook: write an extractive handoff note (no model calls); implies --handoff")
    st.add_argument("--remove-triggers", action="store_true")
    st.add_argument("--doc-trust", choices=["high", "low"],
                    help="high: docs are reliable, answer from them; low (default): docs are hints, always check code")
    rf = sub.add_parser("refresh"); rf.add_argument("domain"); rf.add_argument("--if-changed", action="store_true")
    ss = sub.add_parser("status"); ss.add_argument("domain", nargs="?")
    cf = sub.add_parser("conflicts", help="doc/code conflicts decided by the user (list, or record one)")
    cf.add_argument("domain"); cf.add_argument("--doc", help="doc path as cited by search, to record a decision")
    cf.add_argument("--decision", choices=["code", "doc", "both", "open"]); cf.add_argument("--note", default="")
    sub.add_parser("download-weights")
    sub.add_parser("mcp", help="MCP server over stdio (claude mcp add defrost -- kev-memory mcp)")
    c = sub.add_parser("claude", help="Claude Code setup: slash commands + MCP registration")
    c.add_argument("action", choices=["install"]); c.add_argument("project", nargs="?", default=".")
    c.add_argument("--user", action="store_true", help="install for all projects (~/.claude/commands, user scope)")
    c.add_argument("--no-mcp", action="store_true", help="only copy the slash commands")
    ho = sub.add_parser("handoff", help="write a session handoff note (what the SessionStart hook shows after /clear)")
    ho.add_argument("--goal", required=True); ho.add_argument("--state", default="")
    ho.add_argument("--decision", action="append", default=[]); ho.add_argument("--next", action="append", default=[])
    ho.add_argument("--file", action="append", default=[]); ho.add_argument("--domain")
    ho.add_argument("--no-index", action="store_true")
    br = sub.add_parser("brief", help="print the latest handoff brief for this directory (hook command; no models)")
    br.add_argument("--domain")
    dp = sub.add_parser("docs-plan", help="doc sections to update for a change (model-free)")
    dp.add_argument("--staged", action="store_true"); dp.add_argument("--commit"); dp.add_argument("--json", action="store_true")
    dp.add_argument("--domain")
    dh = sub.add_parser("docs-hook", help="Claude Code hook handler (reads the hook JSON on stdin)")
    dh.add_argument("event", choices=["stop", "commit"])
    dr = sub.add_parser("docs-record", help="git post-commit: save the commit's doc plan as a pending task")
    dr.add_argument("commit", nargs="?", default="HEAD")
    sub.add_parser("docs-pending", help="print pending doc tasks for this repo (SessionStart hook)")
    dv = sub.add_parser("docs-resolve", help="mark a commit's doc task done"); dv.add_argument("commit", nargs="?")
    da = sub.add_parser("docs-auto", help="run claude -p /document-changes <sha> (budget-capped; edits docs only)")
    da.add_argument("commit", nargs="?", default="HEAD"); da.add_argument("--budget", type=float, default=0.5)
    da.add_argument("--dry-run", action="store_true")
    da.add_argument("--merge", action="store_true", help="fast-forward the branch into the checkout when clean")
    sub.add_parser("compact-handoff", help="Claude Code PreCompact hook handler (reads the hook JSON on stdin)")
    cx = sub.add_parser("context", help="the project's context repository (git-backed working memory)")
    cx.add_argument("action", choices=["init", "check", "log", "defrag", "branches", "merge", "remote", "brief"])
    cx.add_argument("arg", nargs="?", help="merge: branch name; remote: URL ('none' to remove)")
    cx.add_argument("--domain"); cx.add_argument("-n", type=int, default=20)
    cx.add_argument("--keep-notes", type=int, default=20, help="defrag: handoff notes kept outside notes/archive")
    cx.add_argument("--review", action="store_true", help="defrag: keep the result on a branch instead of merging")
    cx.add_argument("--repo", help="branches/merge: a project repo instead of the context repo (docs-auto branches)")
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
    elif a.cmd == "conflicts":
        from kev_memory import conflicts
        if a.doc and a.decision:
            print(json.dumps(conflicts.record(a.domain, a.doc, a.decision, note=a.note), indent=1))
        else:
            for r in conflicts.load(a.domain):
                print(f"{r['at']}  {r['doc_path']}:{r['doc_lines']}  {r['meaning']}  {r['note']}")
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
                    a.remove_triggers, doc_trust=a.doc_trust, handoff=a.handoff or a.handoff_on_compact,
                    handoff_on_compact=a.handoff_on_compact, docs_auto_merge=a.docs_auto_merge,
                    docs_sync=a.docs_sync or a.docs_auto, docs_auto=a.docs_auto, docs_budget=a.docs_budget)
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
    elif a.cmd == "handoff":
        from kev_memory import notes
        name = a.domain or notes.domain_for(Path.cwd())
        if not name:
            print("no memory domain for this directory (run kev-memory setup . first, or pass --domain)")
            return 1
        print(notes.write_handoff(name, a.goal, a.state, a.decision, a.next, a.file))
        if not a.no_index:
            print(json.dumps(notes.index(name, wait=False)))
    elif a.cmd == "brief":
        from kev_memory import notes
        text = notes.brief(a.domain or notes.domain_for(Path.cwd()))
        if text:
            print(text)
    elif a.cmd == "docs-plan":
        from kev_memory import docsync
        p = docsync.plan(Path.cwd(), staged=a.staged, commit=a.commit, domain=a.domain)
        print(json.dumps(p, indent=1) if a.json else (docsync.render(p, 25) or "docs up to date for this change"))
    elif a.cmd == "docs-hook":
        from kev_memory import docsync
        try:
            payload = json.loads(sys.stdin.read() or "{}")
            out = docsync.hook(a.event, payload)
        except Exception as e:                           # noqa: BLE001  a broken hook must never block the agent
            print(f"defrost-ai docs hook error: {e}", file=sys.stderr)
            return 0
        if out:
            print(json.dumps(out))
    elif a.cmd == "docs-record":
        from kev_memory import docsync
        f = docsync.record_commit(Path.cwd(), a.commit)
        print(time.strftime("%F %T"), f"docs follow-up recorded: {f}" if f else "no doc follow-up for this commit")
    elif a.cmd == "docs-pending":
        from kev_memory import docsync, notes
        text = docsync.pending_brief(notes.domain_for(Path.cwd()))
        if text:
            print(text)
    elif a.cmd == "docs-resolve":
        from kev_memory import docsync, notes
        print(docsync.resolve(notes.domain_for(Path.cwd()), a.commit), "task(s) resolved")
    elif a.cmd == "docs-auto":
        import os
        import subprocess as sp
        from kev_memory import docsync, notes, worktree
        root = notes.git_root(Path.cwd())
        sha = docsync.git(root, "rev-parse", "--short", a.commit).strip()
        domain = notes.domain_for(root) or ""
        if not any(t["source"] == sha for t in docsync.pending(domain)):
            print(time.strftime("%F %T"), f"{sha}: no pending doc task; nothing to run")
            return 0
        cmd = docsync.auto_command(sha, a.budget)
        branch = f"defrost/docs/{sha}"
        print(time.strftime("%F %T"), "would run" if a.dry_run else "running", f"in a worktree on {branch}:",
              " ".join(cmd), flush=True)
        if a.dry_run:
            return 0
        env = {**os.environ, "KEV_MEMORY_DOMAIN": domain}            # the worktree path is not a registered domain
        res = worktree.run(root, f"docs/{sha}", lambda d: sp.run(cmd, cwd=d, env=env, check=False),
                           message=f"docs: document {sha} (defrost-ai docs-auto)", merge=a.merge, branch=branch)
        print(time.strftime("%F %T"), json.dumps(res), flush=True)
        return 0 if res["status"] != "failed" else 1
    elif a.cmd == "compact-handoff":
        from kev_memory import compact_handoff
        try:
            f = compact_handoff.run(json.loads(sys.stdin.read() or "{}"))
        except Exception as e:                           # noqa: BLE001  a broken hook must never block compaction
            print(f"defrost-ai compact handoff error: {e}", file=sys.stderr)
            return 0
        if f:
            print(f"defrost-ai: handoff note written before compaction: {f}", file=sys.stderr)
    elif a.cmd == "context":
        from kev_memory import context_repo as cr, notes, worktree
        name = a.domain or notes.domain_for(Path.cwd())
        if not name and not (a.repo and a.action in ("branches", "merge")):
            print("no memory domain for this directory (run kev-memory setup . first, or pass --domain)")
            return 1
        if a.action == "init":
            print(cr.ensure(name)); print(cr.register(name))
        elif a.action == "check":
            errors = cr.check(name)
            print("\n".join(errors) or "context repository OK")
            return 1 if errors else 0
        elif a.action == "log":
            for r in cr.log(name, a.n):
                print(f"{r['sha']}  {r['date']}  {r['author']:<12} {r['subject']}")
        elif a.action == "brief":
            print(cr.map_brief(name))
        elif a.action == "defrag":
            print(json.dumps(cr.defrag(name, a.keep_notes, merge=not a.review), indent=1))
        elif a.action == "branches":
            for b in worktree.branches(Path(a.repo).resolve() if a.repo else cr.repo_dir(name)):
                print(f"{b['branch']:<45} {b['sha']}  {b['date']}  +{b['ahead']}  {b['subject']}")
        elif a.action == "merge":
            if not a.arg:
                print("usage: kev-memory context merge <branch> [--repo PATH]")
                return 1
            res = worktree.merge_branch(Path(a.repo).resolve() if a.repo else cr.repo_dir(name), a.arg)
            print(json.dumps(res, indent=1))
            return 0 if res["status"] == "merged" else 1
        elif a.action == "remote":
            url = None if (a.arg or "none") == "none" else a.arg
            print(json.dumps(cr.set_remote(name, url), indent=1))
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
