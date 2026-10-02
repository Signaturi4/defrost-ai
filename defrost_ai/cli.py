"""defrost: your project's docs and code, searchable by your AI tools, with citations.

Start here:
  defrost setup                 set up this repository (asks 3 questions; --yes takes the recommended answers)
  defrost search "question"     find the doc sections that answer it, with the code they name
  defrost status                what is indexed, how fresh it is, which settings are active

Day to day:
  defrost refresh               update the index now (setup already does this after every merge to main)
  defrost docs [FILES]          which doc sections to update for your change
  defrost note "goal"           save a handoff note for the next session; --brief shows the latest
  defrost config [KEY [VALUE]]  personal settings, e.g. `defrost config search.mode fast`

Search modes:
  accurate (default)  best results; reranks when the two retrievers disagree (~1-2 s on Apple Silicon)
  fast                no reranker (~0.1 s); a little less accurate. Use --fast, or make it your default.

For tools and scripts: defrost mcp (MCP server), defrost serve (local HTTP service)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

PUBLIC = ["setup", "search", "status", "refresh", "docs", "note", "config", "mcp", "serve"]
PROFILES = {
    "minimal":  dict(on_main_merge=True),
    "standard": dict(on_main_merge=True, doc_rules=True, handoff=True, docs_sync=True, docs_gate=False),
    "full":     dict(on_main_merge=True, doc_rules=True, handoff=True, docs_sync=True, docs_gate=True,
                     handoff_on_compact=True),
}
PROFILE_TEXT = {
    "minimal": "keep the index fresh after every merge or commit to main; nothing else",
    "standard": "minimal + doc-writing rules in CLAUDE.md + handoff notes after /clear + a reminder of commits "
                "whose docs need an update (recommended)",
    "full": "standard + Claude is asked to update docs before it finishes and before it commits + a handoff note "
            "is written automatically before compaction",
}


def _k(v):
    return v if v == "auto" else int(v)


def _version_line() -> str:
    from defrost_ai import __version__
    from defrost_ai.models.weights import WEIGHTS_VERSION
    return f"defrost {__version__} (weights {WEIGHTS_VERSION})"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="defrost", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=_version_line())
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    st = sub.add_parser("setup", help="set up a repository (quick: 3 questions)",
                        description="Index this repository and keep it fresh. Without options in a terminal it asks "
                                    "3 questions; --yes takes the recommended answers.")
    st.add_argument("path", nargs="?", default=".")
    st.add_argument("--profile", choices=list(PROFILES), help="; ".join(f"{k}: {v}" for k, v in PROFILE_TEXT.items()))
    st.add_argument("--mode", choices=["accurate", "fast"], help="your default search mode (saved in config)")
    st.add_argument("--doc-trust", choices=["low", "high"],
                    help="low (default): docs are hints, Claude checks the code. high: docs are reliable.")
    st.add_argument("--yes", "-y", action="store_true", help="no questions: recommended answers")
    st.add_argument("--remove", "--remove-triggers", dest="remove_triggers", action="store_true",
                    help="remove every hook and schedule defrost installed in this repository")
    adv = st.add_argument_group("customize (each overrides the profile)")
    adv.add_argument("--domain", help="memory name (default: the folder name)")
    adv.add_argument("--build", choices=["now", "background", "skip"], default="now")
    adv.add_argument("--every-hours", type=float, help="also refresh on a schedule (launchd on macOS, cron on Linux)")
    adv.add_argument("--claude-hook", action="store_true", help="also refresh when a Claude session starts")
    adv.add_argument("--no-doc-rules", action="store_true", help="do not add the doc-writing rules to CLAUDE.md")
    adv.add_argument("--handoff-on-compact", action="store_true", help="write a handoff note before compaction")
    adv.add_argument("--docs-auto", action="store_true",
                     help="write docs with claude -p after commits made outside Claude (costs Claude usage)")
    adv.add_argument("--docs-budget", type=float, default=0.5, help="USD cap per automatic docs run")
    adv.add_argument("--docs-auto-merge", action="store_true", help="merge automatic docs without review")
    adv.add_argument("--memory-dir", default="defrost-memory", help="notes folder inside the project")
    adv.add_argument("--memory-home", action="store_true", help="keep notes in ~/.defrost-ai, not in the project")
    adv.add_argument("--claude", action="store_true", help="also install slash commands + MCP for this project only")
    for flag in ("--on-main-merge", "--doc-rules", "--docs-sync", "--handoff"):     # before 1.2: pick features one by one
        st.add_argument(flag, action="store_true", help=argparse.SUPPRESS)

    s = sub.add_parser("search", help="find the doc sections that answer a question")
    s.add_argument("query")
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--fast", dest="mode", action="store_const", const="fast", help="no reranker (~0.1 s)")
    mode.add_argument("--accurate", dest="mode", action="store_const", const="accurate", help="best results (default)")
    mode.add_argument("--mode", dest="mode", help=argparse.SUPPRESS)          # expert: bm25|dense|hybrid|rerank|all
    s.add_argument("-k", type=_k, default=None, help='sections to return: a number or "auto" (default: config)')
    s.add_argument("--domain", action="append", help="search these memories (default: all)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--merge", choices=["rerank", "rrf"], help=argparse.SUPPRESS)
    s.add_argument("--local", action="store_true", help=argparse.SUPPRESS)

    ss = sub.add_parser("status", help="what is indexed, how fresh, which settings")
    ss.add_argument("domain", nargs="?"); ss.add_argument("--json", action="store_true")

    rf = sub.add_parser("refresh", help="update the index now (incremental)")
    rf.add_argument("domain", nargs="?", help="memory name (default: this directory's)")
    rf.add_argument("--if-changed", action="store_true", help="skip when no doc or code file changed")
    rf.add_argument("--rollback", action="store_true", help="restore the previous build instead")

    dc = sub.add_parser("docs", help="doc sections to update for a change")
    dc.add_argument("files", nargs="*", help="changed files (default: your uncommitted change)")
    dc.add_argument("--staged", action="store_true"); dc.add_argument("--commit")
    dc.add_argument("--pending", action="store_true", help="commits whose docs still need an update")
    dc.add_argument("--done", metavar="SHA", help="mark a commit's doc follow-up as done")
    dc.add_argument("--json", action="store_true")

    nt = sub.add_parser("note", help="handoff notes and doc/code decisions")
    nt.add_argument("goal", nargs="?", help="what the work is for (with acceptance criteria)")
    nt.add_argument("--state", default=""); nt.add_argument("--next", action="append", default=[])
    nt.add_argument("--why", action="append", default=[], help="a decision and its reason (repeatable)")
    nt.add_argument("--file", action="append", default=[])
    nt.add_argument("--brief", action="store_true", help="show the latest note (what a new session starts from)")
    nt.add_argument("--history", action="store_true", help="every note and decision, newest first")
    nt.add_argument("--conflict", metavar="DOC_PATH", help="record your decision on a doc/code conflict")
    nt.add_argument("--verdict", choices=["code", "doc", "both", "open"],
                    help="code: the code is right | doc: the doc is right | both: no conflict | open: unsure")
    nt.add_argument("--domain")

    cg = sub.add_parser("config", help="personal settings (search mode, doc trust, models, port)")
    cg.add_argument("key", nargs="?"); cg.add_argument("value", nargs="?")
    cg.add_argument("--reset", action="store_true", help="back to the default")

    sub.add_parser("mcp", help="MCP server (stdio): claude mcp add defrost -- defrost mcp")
    v = sub.add_parser("serve", help="local HTTP service (started automatically)")
    v.add_argument("--port", type=int); v.add_argument("--host", default="127.0.0.1")

    hk = sub.add_parser("hook")                                   # hidden: the one entry point every hook calls
    hk.add_argument("event", choices=["brief", "pending", "stop", "commit", "post-commit", "pre-compact"])

    # ---- old commands (hidden; kept so installed hooks and scripts keep working) ------------------------------------
    b = sub.add_parser("build"); b.add_argument("workspace"); b.add_argument("--domain"); b.add_argument("--description", default="")
    u = sub.add_parser("update"); u.add_argument("domain")
    r = sub.add_parser("rollback"); r.add_argument("domain")
    sub.add_parser("domains")
    df = sub.add_parser("docs-for")
    df.add_argument("paths", nargs="+"); df.add_argument("--domain", action="append")
    e = sub.add_parser("benchmark"); e.add_argument("--suite", required=True); e.add_argument("--memory", required=True)
    e.add_argument("--split", default="dev"); e.add_argument("--out"); e.add_argument("--rankings")
    e.add_argument("--only", help="keep suite rows whose \"suite\" field matches (multi-memory suites)")
    sub.add_parser("verify-weights")
    cf = sub.add_parser("conflicts")
    cf.add_argument("domain"); cf.add_argument("--doc", help="doc path as cited by search, to record a decision")
    cf.add_argument("--decision", choices=["code", "doc", "both", "open"]); cf.add_argument("--note", default="")
    sub.add_parser("download-weights")
    c = sub.add_parser("claude")
    c.add_argument("action", choices=["install"]); c.add_argument("project", nargs="?", default=".")
    c.add_argument("--user", action="store_true", help="install for all projects (~/.claude/commands, user scope)")
    c.add_argument("--no-mcp", action="store_true", help="only copy the slash commands")
    ho = sub.add_parser("handoff")
    ho.add_argument("--goal", required=True); ho.add_argument("--state", default="")
    ho.add_argument("--decision", action="append", default=[]); ho.add_argument("--next", action="append", default=[])
    ho.add_argument("--file", action="append", default=[]); ho.add_argument("--domain")
    ho.add_argument("--no-index", action="store_true")
    br = sub.add_parser("brief")
    br.add_argument("--domain")
    dp = sub.add_parser("docs-plan")
    dp.add_argument("--staged", action="store_true"); dp.add_argument("--commit"); dp.add_argument("--json", action="store_true")
    dp.add_argument("--domain")
    dh = sub.add_parser("docs-hook")
    dh.add_argument("event", choices=["stop", "commit"])
    dr = sub.add_parser("docs-record")
    dr.add_argument("commit", nargs="?", default="HEAD")
    sub.add_parser("docs-pending")
    dv = sub.add_parser("docs-resolve"); dv.add_argument("commit", nargs="?")
    da = sub.add_parser("docs-auto")
    da.add_argument("commit", nargs="?", default="HEAD"); da.add_argument("--budget", type=float, default=0.5)
    da.add_argument("--dry-run", action="store_true")
    da.add_argument("--merge", action="store_true", help="fast-forward the branch into the checkout when clean")
    sub.add_parser("compact-handoff")
    cx = sub.add_parser("context")
    cx.add_argument("action", choices=["init", "check", "log", "defrag", "branches", "merge", "remote", "brief",
                                             "where", "place"])
    cx.add_argument("arg", nargs="?", help="merge: branch name; remote: URL ('none' to remove); "
                                           "place: project path ('home' moves it back to ~/.defrost-ai)")
    cx.add_argument("--dir", default="defrost-memory", help="place: folder name inside the project")
    cx.add_argument("--domain"); cx.add_argument("-n", type=int, default=20)
    cx.add_argument("--keep-notes", type=int, default=20, help="defrag: handoff notes kept outside notes/archive")
    cx.add_argument("--review", action="store_true", help="defrag: keep the result on a branch instead of merging")
    cx.add_argument("--repo", help="branches/merge: a project repo instead of the context repo (docs-auto branches)")
    a = ap.parse_args(argv)
    if a.cmd in NEW:
        return NEW[a.cmd](a) or 0

    if a.cmd == "build":
        from defrost_ai.builder import build
        from defrost_ai.config import Workspace
        from defrost_ai.library import register
        build(a.workspace)
        if a.domain:
            register(a.domain, a.workspace, a.description)
            print(f"registered domain {a.domain!r}")
    elif a.cmd == "update":
        from defrost_ai.builder import build
        from defrost_ai.library import read_registry
        build(read_registry()["domains"][a.domain]["workspace"])
    elif a.cmd == "rollback":
        from defrost_ai.builder import rollback
        from defrost_ai.library import read_registry
        print("rolled back" if rollback(read_registry()["domains"][a.domain]["workspace"]) else "no previous build")
    elif a.cmd == "domains":
        from defrost_ai.library import Library
        print(json.dumps(Library().domains(), indent=1))
    elif a.cmd == "conflicts":
        from defrost_ai import conflicts
        if a.doc and a.decision:
            print(json.dumps(conflicts.record(a.domain, a.doc, a.decision, note=a.note), indent=1))
        else:
            for r in conflicts.load(a.domain):
                print(f"{r['at']}  {r['doc_path']}:{r['doc_lines']}  {r['meaning']}  {r['note']}")
    elif a.cmd == "docs-for":
        from defrost_ai.library import Library
        hits = Library().docs_for(a.paths, a.domain)
        for h in hits:
            print(f"{h['file']}: {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}")
        if not hits:
            print("no doc section links to these files")
    elif a.cmd == "benchmark":
        from defrost_ai.evaluation.benchmark import print_report, run
        res = run(a.suite, a.memory, a.split, save_rankings=a.rankings, only=a.only)
        print_report(res)
        if a.out:
            open(a.out, "w").write(json.dumps(res, indent=1))
    elif a.cmd == "download-weights":
        from defrost_ai.models.merged_cache import gc_current
        from defrost_ai.models.weights import WEIGHTS_URL, WEIGHTS_VERSION, download_weights, models_dir
        if _weights_ok():
            print(f"weights ready: {models_dir(download=False)}")
            gc_current()
            return 0
        try:
            download_weights(log=lambda m: print(m, flush=True))
        except Exception as e:                           # noqa: BLE001  (a clear message instead of a traceback)
            import urllib.error
            releases = "https://github.com/Signaturi4/defrost-ai/releases"
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                print(f"defrost: weights v{WEIGHTS_VERSION} are not published ({WEIGHTS_URL} returned 404).\n"
                      f"  See {releases} for available weights; set DEFROST_MODELS to a local weights folder.",
                      file=sys.stderr)
            elif isinstance(e, urllib.error.URLError):
                print(f"defrost: could not download the weights ({e.reason}). Check the network and retry: "
                      f"defrost download-weights", file=sys.stderr)
            else:
                print(f"defrost: weights download failed: {e}", file=sys.stderr)
            return 2
        gc_current()                                     # merged caches of older weights
    elif a.cmd == "mcp":
        from defrost_ai.service.mcp_server import main as mcp_main
        mcp_main()
    elif a.cmd == "claude":
        from defrost_ai.integrations.claude import install
        for line in install(Path(a.project), user=a.user, register_mcp=not a.no_mcp):
            print(line)
    elif a.cmd == "handoff":
        from defrost_ai import notes
        name = a.domain or notes.domain_for(Path.cwd())
        if not name:
            print("no memory domain for this directory (run defrost setup . first, or pass --domain)")
            return 1
        print(notes.write_handoff(name, a.goal, a.state, a.decision, a.next, a.file))
        if not a.no_index:
            print(json.dumps(notes.index(name, wait=False)))
    elif a.cmd == "brief":
        from defrost_ai import notes
        text = notes.brief(a.domain or notes.domain_for(Path.cwd()))
        if text:
            print(text)
    elif a.cmd == "docs-plan":
        from defrost_ai import docsync
        p = docsync.plan(Path.cwd(), staged=a.staged, commit=a.commit, domain=a.domain)
        print(json.dumps(p, indent=1) if a.json else (docsync.render(p, 25) or "docs up to date for this change"))
    elif a.cmd == "docs-hook":
        from defrost_ai import docsync
        try:
            payload = json.loads(sys.stdin.read() or "{}")
            out = docsync.hook(a.event, payload)
        except Exception as e:                           # noqa: BLE001  a broken hook must never block the agent
            print(f"defrost-ai docs hook error: {e}", file=sys.stderr)
            return 0
        if out:
            print(json.dumps(out))
    elif a.cmd == "docs-record":
        from defrost_ai import docsync
        f = docsync.record_commit(Path.cwd(), a.commit)
        print(time.strftime("%F %T"), f"docs follow-up recorded: {f}" if f else "no doc follow-up for this commit")
    elif a.cmd == "docs-pending":
        from defrost_ai import docsync, notes
        text = docsync.pending_brief(notes.domain_for(Path.cwd()))
        if text:
            print(text)
    elif a.cmd == "docs-resolve":
        from defrost_ai import docsync, notes
        print(docsync.resolve(notes.domain_for(Path.cwd()), a.commit), "task(s) resolved")
    elif a.cmd == "docs-auto":
        import os
        import subprocess as sp
        from defrost_ai import docsync, notes, worktree
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
        env = {**os.environ, "DEFROST_DOMAIN": domain}            # the worktree path is not a registered domain
        res = worktree.run(root, f"docs/{sha}", lambda d: sp.run(cmd, cwd=d, env=env, check=False),
                           message=f"docs: document {sha} (defrost-ai docs-auto)", merge=a.merge, branch=branch)
        print(time.strftime("%F %T"), json.dumps(res), flush=True)
        return 0 if res["status"] != "failed" else 1
    elif a.cmd == "compact-handoff":
        from defrost_ai import compact_handoff
        try:
            f = compact_handoff.run(json.loads(sys.stdin.read() or "{}"))
        except Exception as e:                           # noqa: BLE001  a broken hook must never block compaction
            print(f"defrost-ai compact handoff error: {e}", file=sys.stderr)
            return 0
        if f:
            print(f"defrost-ai: handoff note written before compaction: {f}", file=sys.stderr)
    elif a.cmd == "context":
        from defrost_ai import context_repo as cr, notes, worktree
        name = a.domain or notes.domain_for(Path.cwd())
        if not name and not (a.repo and a.action in ("branches", "merge")):
            print("no memory domain for this directory (run defrost setup . first, or pass --domain)")
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
                print("usage: defrost context merge <branch> [--repo PATH]")
                return 1
            res = worktree.merge_branch(Path(a.repo).resolve() if a.repo else cr.repo_dir(name), a.arg)
            print(json.dumps(res, indent=1))
            return 0 if res["status"] == "merged" else 1
        elif a.action == "where":
            print(cr.repo_dir(name))
        elif a.action == "place":
            target = None if a.arg == "home" else Path(a.arg or ".").resolve()
            print(json.dumps(cr.place(name, target, a.dir), indent=1))
        elif a.action == "remote":
            url = None if (a.arg or "none") == "none" else a.arg
            print(json.dumps(cr.set_remote(name, url), indent=1))
    elif a.cmd == "verify-weights":
        from defrost_ai.models.weights import models_dir, verify
        res = verify()
        print(json.dumps({"dir": str(models_dir()), **res}, indent=1))
        return 0 if res["ok"] else 1
    return 0


# ---- commands ------------------------------------------------------------------------------------------------------
def _here(a=None) -> str | None:
    from defrost_ai import notes
    return getattr(a, "domain", None) if isinstance(getattr(a, "domain", None), str) else notes.domain_for(Path.cwd())


def _ask(question: str, options: list[tuple[str, str]], default: int = 0) -> str:
    print(f"\n{question}")
    for i, (value, text) in enumerate(options, 1):
        print(f"  {i}. {value:<9} {text}{'  (recommended)' if i - 1 == default else ''}")
    while True:
        raw = input(f"Choose 1-{len(options)} [{default + 1}]: ").strip()
        if not raw:
            return options[default][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        if raw in {v for v, _ in options}:
            return raw
        print("  please type a number from the list")


def cmd_setup(a):
    from defrost_ai import settings
    from defrost_ai.project_setup import setup
    if a.remove_triggers:
        print(f"removed defrost hooks and schedules from {Path(a.path).resolve()}")
        setup(a.path, a.domain, remove=True, log=lambda m: None)
        return 0
    interactive = sys.stdin.isatty() and not a.yes and not (a.profile or a.mode or a.doc_trust)
    if interactive:
        print("defrost setup: 3 questions. Press Enter for the recommended answer.")
        a.profile = _ask("How much should defrost do in this repository?",
                         [(k, v) for k, v in PROFILE_TEXT.items()], default=1)
        a.mode = _ask("Default search mode?",
                      [("accurate", "best results; ~1-2 s when the reranker runs"),
                       ("fast", "~0.1 s; no reranker, a little less accurate")], default=0)
        a.doc_trust = _ask("How much should Claude trust this project's docs?",
                           [("low", "code is the truth: docs are hints, Claude checks the code (fast-changing projects)"),
                            ("high", "docs are reliable: answer from them, check code only when flagged")], default=0)
    legacy = {k: True for k in ("on_main_merge", "doc_rules", "docs_sync", "handoff") if getattr(a, k)}
    profile = a.profile or ("custom" if legacy else "standard")
    opts = dict(PROFILES.get(profile, legacy))
    if a.no_doc_rules:
        opts["doc_rules"] = False
    if a.handoff_on_compact:
        opts.update(handoff=True, handoff_on_compact=True)
    if a.docs_auto:
        opts.update(docs_sync=True, docs_auto=True)
    if a.mode:
        settings.set_("search.mode", a.mode)
    print(f"\nSetting up {Path(a.path).resolve()} (profile {profile}: {PROFILE_TEXT.get(profile, ', '.join(legacy))})")
    res = setup(a.path, a.domain, a.build, every_hours=a.every_hours, claude_hook=a.claude_hook, claude=a.claude,
                doc_trust=a.doc_trust or settings.get("project.doc_trust"), docs_budget=a.docs_budget,
                docs_auto_merge=a.docs_auto_merge, memory_dir=None if a.memory_home else a.memory_dir,
                log=lambda m: print("  " + m, flush=True), **opts)
    _print_setup(res, settings.get("search.mode"))
    return 0 if (res.get("build") or {}).get("job", "done") != "failed" else 1


def _print_setup(res: dict, mode: str) -> None:
    b = res.get("build") or {}
    trig = res.get("triggers", {})
    labels = {"on_main_merge": "refresh after every merge or commit to main", "every_hours": "refresh on a schedule",
              "claude_hook": "refresh when a Claude session starts", "handoff": "handoff note shown after /clear",
              "handoff_on_compact": "handoff note written before compaction",
              "docs_sync": "doc follow-ups for every commit"}
    from defrost_ai.project_setup import status
    c = next((r.get("counts") or {} for r in status(res["domain"])), {})
    built = (f": {c.get('docs', 0)} docs, {c.get('sections', 0)} sections, {c.get('doc_code_links', 0)} doc->code links"
             if b.get("refreshed") else "")
    print(f"\nDone. Memory '{res['domain']}'{built}")
    for k in trig:
        print(f"  - {labels.get(k, k)}")
    print(f"  - doc trust: {res.get('doc_trust')}; default search mode: {mode}")
    print("\nTry:  defrost search \"how does <something> work?\"      In Claude: just ask; it calls the memory.")
    print("Change later:  defrost setup --profile full | defrost config search.mode fast | defrost setup --remove")


def cmd_search(a):
    from defrost_ai import settings
    from defrost_ai.memory import Memory
    k = a.k if a.k is not None else _k(str(settings.get("search.k")))
    if a.local:                                      # load the models in this process (slow: every call)
        from defrost_ai.library import Library
        res = Library().search(a.query, a.domain, a.mode, k, a.merge)
    else:                                            # resident service: models stay loaded between calls
        from defrost_ai.service import client
        client.ensure_service()
        res = client.search(a.query, a.domain, a.mode, k, context=False, merge=a.merge)
    if a.json:
        print(json.dumps(res, indent=1))
        return 0
    used = res["mode_used"] if isinstance(res["mode_used"], str) else ", ".join(sorted(set(res["mode_used"].values())))
    note = {"fast": "  (fast: no reranker; use --accurate for the best results)",
            "accurate": f"  (accurate: {'reranked' if 'rerank' in used else 'retrievers agreed, no rerank needed'})"}
    print(f"{len(res['hits'])} sections{note.get(res['mode'], '')}\n\n" + Memory.context(res))


def cmd_status(a):
    from defrost_ai import settings
    from defrost_ai.project_setup import status
    rows = status(a.domain)
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    if not rows:
        print("No memories yet. Run `defrost setup` in a repository.")
    names = {r["domain"] for r in rows}
    notes_of = {r["domain"][:-len("-context")]: r for r in rows
                if r["domain"].endswith("-context") and r["domain"][:-len("-context")] in names}
    for r in rows:
        if r["domain"].endswith("-context") and r["domain"][:-len("-context")] in names:
            continue                                     # shown under its project
        c = r.get("counts") or {}
        fresh = "stale: " + r["why"] if r["stale"] else "up to date"
        print(f"{r['domain']}: {c.get('sections', 0)} sections, {c.get('doc_code_links', 0)} doc->code links; "
              f"built {str(r.get('built_at') or 'never')[:16]}; {fresh}")
        if r.get("triggers"):
            print(f"  refresh: {', '.join(r['triggers'])}")
        n = notes_of.get(r["domain"])
        if n:
            print(f"  notes and decisions: {(n.get('counts') or {}).get('docs', 0)} files")
    print(f"\nsearch mode {settings.get('search.mode')} | backend {_backend()} | settings: defrost config")
    print(_weights_line())
    print(_mcp_line())


def _mcp_line() -> str:
    from defrost_ai.service.mcp_server import mcp_available
    if mcp_available():
        return "MCP server: ready (`defrost mcp`)"
    return "MCP server: NOT available, the `mcp` package is missing (run `defrost mcp` for the fix)"


def _backend() -> str:
    try:
        from defrost_ai.models import mlx_backend
        return "mlx" if mlx_backend.available() else "torch"
    except Exception:                                   # noqa: BLE001
        return "torch"


def _weights_line() -> str:
    from defrost_ai.models.merged_cache import merged_cache_size
    from defrost_ai.models.weights import status
    w = status()
    if w["dir"] is None:
        line = f"weights: not downloaded (expected v{w['expected']}; run `defrost download-weights`)"
    elif w["matches"]:
        line = f"weights: v{w['version']} ({w['source']}), matches the pinned v{w['expected']}"
    else:
        line = (f"weights: v{w['version']} ({w['source']}), MISMATCH: expected v{w['expected']} "
                f"(run `defrost download-weights`)")
    n, size = merged_cache_size()
    return line + f"\nmerged-weights cache: {n} model(s), {size / 1e9:.1f} GB"


def cmd_refresh(a):
    from defrost_ai.project_setup import refresh
    name = a.domain or _here()
    if not name:
        print("This directory has no memory yet. Run `defrost setup` here first.")
        return 1
    if a.rollback:
        from defrost_ai.builder import rollback
        from defrost_ai.library import read_registry
        print("rolled back to the previous build" if rollback(read_registry()["domains"][name]["workspace"])
              else "no previous build to roll back to")
        return 0
    res = refresh(name, a.if_changed, log=lambda m: print(time.strftime("%F %T"), m, flush=True))
    return 0 if res.get("job") != "failed" else 1


def cmd_docs(a):
    from defrost_ai import docsync, notes
    domain = _here()
    if a.pending:
        print(docsync.pending_brief(domain) or "no commits waiting for doc updates")
    elif a.done:
        print(docsync.resolve(domain, a.done), "follow-up(s) marked done")
    elif a.files:
        from defrost_ai.library import Library
        hits = Library().docs_for(a.files, [domain] if domain else None)
        for h in hits:
            print(f"{h['file']}: {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}")
        print("" if hits else "no doc section describes these files")
    else:
        p = docsync.plan(Path.cwd(), staged=a.staged, commit=a.commit, domain=domain)
        print(json.dumps(p, indent=1) if a.json else (docsync.render(p, 25) or "docs are up to date for this change"))


def cmd_note(a):
    from defrost_ai import conflicts, notes
    name = a.domain or notes.domain_for(Path.cwd())
    if not name:
        print("This directory has no memory yet. Run `defrost setup` here first.")
        return 1
    if a.conflict:
        if not a.verdict:
            print("add --verdict code|doc|both|open")
            return 1
        row = conflicts.record(name, a.conflict, a.verdict)
        print(f"decision saved: {a.conflict}: {row['meaning']}")
    elif a.brief:
        print(notes.brief(name) or "no notes yet")
    elif a.history:
        from defrost_ai import context_repo
        for r in context_repo.log(name, 30):
            print(f"{r['date']}  {r['subject']}")
    elif a.goal:
        f = notes.write_handoff(name, a.goal, a.state, a.why, a.next, a.file)
        print(f"note saved: {f}")
        try:
            notes.index(name, wait=False)
        except Exception:                                # noqa: BLE001  (searchable after the next refresh)
            pass
    else:
        print('usage: defrost note "goal" [--state ...] [--next ...] | --brief | --history | --conflict DOC --verdict X')
        return 1


def cmd_config(a):
    from defrost_ai import settings
    if not a.key:
        print(settings.show())
        return 0
    if a.key not in settings.BY_KEY:
        print(f"unknown setting {a.key!r}. Settings:\n" + "\n".join(f"  {k}" for k in settings.BY_KEY))
        return 1
    if a.reset or a.value is not None:
        try:
            settings.set_(a.key, None if a.reset else a.value)
        except ValueError as e:
            print(e)
            return 1
    print(f"{a.key} = {settings.get(a.key)} ({settings.source(a.key)})")


def cmd_mcp(a):
    from defrost_ai.service.mcp_server import main as mcp_main
    mcp_main()


def cmd_serve(a):
    from defrost_ai import settings
    from defrost_ai.service.http_server import serve
    serve(a.port or settings.get("service.port"), a.host)


def cmd_hook(a):
    """Every installed hook calls `defrost hook <event>`; a failing hook prints to stderr and never blocks."""
    try:
        from defrost_ai import compact_handoff, docsync, notes
        if a.event == "brief":
            text = notes.brief(notes.domain_for(Path.cwd()))
        elif a.event == "pending":
            text = docsync.pending_brief(notes.domain_for(Path.cwd()))
        elif a.event in ("stop", "commit"):
            out = docsync.hook(a.event, json.loads(sys.stdin.read() or "{}"))
            text = json.dumps(out) if out else ""
        elif a.event == "post-commit":
            f = docsync.record_commit(Path.cwd(), "HEAD")
            text = time.strftime("%F %T ") + (f"docs follow-up recorded: {f}" if f else "no doc follow-up")
        else:                                            # pre-compact
            f = compact_handoff.run(json.loads(sys.stdin.read() or "{}"))
            print(f"defrost: handoff note written before compaction: {f}" if f else "", file=sys.stderr)
            text = ""
        if text:
            print(text)
    except Exception as e:                               # noqa: BLE001
        print(f"defrost hook {a.event}: {e}", file=sys.stderr)
    return 0


NEW = {"setup": cmd_setup, "search": cmd_search, "status": cmd_status, "refresh": cmd_refresh, "docs": cmd_docs,
       "note": cmd_note, "config": cmd_config, "mcp": cmd_mcp, "serve": cmd_serve, "hook": cmd_hook}


def _weights_ok() -> bool:
    try:
        from defrost_ai.models.weights import CACHE, WEIGHTS_VERSION, models_dir, verify
        d = models_dir(download=False)
        return verify(d)["ok"] and (d != CACHE / "models" or verify(d)["version"] == WEIGHTS_VERSION)
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
