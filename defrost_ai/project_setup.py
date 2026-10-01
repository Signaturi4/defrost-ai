"""One-shot project setup and refresh triggers.

    defrost setup [PATH] [--domain NAME] [--build now|background|skip]
                     [--on-main-merge] [--every-hours N] [--claude-hook] [--doc-rules] [--claude]
    defrost refresh DOMAIN [--if-changed]      what every trigger runs (cheap no-op when nothing changed)
    defrost status [DOMAIN]                     staleness of each domain and its installed triggers
    defrost setup [PATH] --remove-triggers

Triggers (any combination):
  on-main-merge   git post-merge + post-commit hooks; act only on main/master, and only when a doc or code file changed
  every-hours N   macOS launchd agent (Linux: a crontab line) that runs `refresh --if-changed` every N hours
  claude-hook     Claude Code SessionStart hook in .claude/settings.json: refresh in the background when stale
All triggers run detached, never block git or Claude, and log to ~/.defrost-ai/<domain>.refresh.log.
The memory itself lives in ~/.defrost-ai/<domain>; nothing but hooks/settings is written into the project."""
from __future__ import annotations

import json
import re
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from defrost_ai.config import CODE_SUFFIXES, DOC_SUFFIXES

HOME = Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser()
MARK_START, MARK_END = "# >>> defrost-ai >>>", "# <<< defrost-ai <<<"
HOOK_TAG = "defrost-ai refresh"


def exe() -> list[str]:
    """Absolute command for defrost, so hooks work outside the venv."""
    found = shutil.which("defrost")
    return [found] if found else [sys.executable, "-m", "defrost_ai.cli"]


def git(root: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def workspace_for(path: Path, name: str) -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    ws = HOME / f"{name}.workspace.json"
    if not ws.exists():
        ws.write_text(json.dumps({"name": name, "out": str(HOME / name), "components": [{
            "name": name, "path": str(path),
            "exclude": ["docs/DOC_RULES.md", "docs/templates", "docs/tools"]}]}, indent=1))   # the doc-rules kit itself
    return ws


def state_file(name: str) -> Path:
    return HOME / f"{name}.triggers.json"


# ---- refresh (what every trigger calls) -----------------------------------------------------------------------------
def changed_since_build(name: str) -> tuple[bool, str]:
    from defrost_ai.library import read_registry
    reg = read_registry()["domains"].get(name)
    if not reg:
        return True, "not built yet"
    ws = json.loads(Path(reg["workspace"]).read_text())
    man_f = Path(ws["out"]).expanduser() / "manifest.json"
    if not man_f.exists():
        return True, "no manifest"
    man = json.loads(man_f.read_text())
    suffixes = tuple(DOC_SUFFIXES | CODE_SUFFIXES)
    built_ts = time.mktime(time.strptime(man["built_at"][:19], "%Y-%m-%dT%H:%M:%S"))
    for comp in ws["components"]:
        root = Path(comp["path"]).expanduser()
        old = man.get("sources", {}).get(str(root))
        head = git(root, "rev-parse", "HEAD")
        if head and old:
            if head == old and not git(root, "status", "--porcelain", "--untracked-files=no"):
                continue
            files = git(root, "diff", "--name-only", old, head).splitlines() if head != old else []
            for f in git(root, "diff", "--name-only").splitlines():   # uncommitted edits: only if newer than build
                if (root / f).exists() and (root / f).stat().st_mtime > built_ts:
                    files.append(f)
            hit = [f for f in files if f.endswith(suffixes) and _indexed(f, comp.get("exclude", []))]
            if hit:
                return True, f"{len(hit)} doc/code file(s) changed in {comp['name']}, e.g. {hit[0]}"
            continue
        for f in root.rglob("*"):                       # not a git repo: newest mtime of a doc/code file
            if f.suffix in suffixes and ".git" not in f.parts and f.stat().st_mtime > built_ts:
                return True, f"{f.relative_to(root)} modified after the last build"
    return False, "up to date"


def _indexed(rel: str, exclude) -> bool:
    """Same skip rules as Component.files: no SKIP_DIRS, no hidden dirs (except .github), no excludes."""
    from defrost_ai.config import SKIP_DIRS
    parts = rel.split("/")
    dirs = parts[:-1]
    if any(d in SKIP_DIRS or (d.startswith(".") and d != ".github") for d in dirs):
        return False
    return not any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in exclude)


def refresh(name: str, if_changed: bool = True, log=print) -> dict:
    stale, why = changed_since_build(name) if if_changed else (True, "forced")
    if not stale:
        log(f"[{name}] {why}; nothing to do")
        return {"domain": name, "refreshed": False, "reason": why}
    from defrost_ai.service import client
    job = client.update(name, wait=True)
    log(f"[{name}] {why} -> {job.get('state')} in {job.get('seconds')}s "
        f"({_n_changed(job.get('manifest', {}).get('changed', {}))} changed docs)")
    return {"domain": name, "refreshed": job.get("state") == "done", "reason": why, "job": job.get("state")}


def _n_changed(ch) -> int:
    if isinstance(ch, dict):
        n = ch.get("n")
        if isinstance(n, int):
            return n
        if isinstance(n, dict):
            return sum(v for v in n.values() if isinstance(v, int))
        return sum(len(v) if isinstance(v, (list, dict)) else v if isinstance(v, int) else 0 for v in ch.values())
    return 0


def detached(name: str) -> str:
    """Shell snippet that runs the refresh in the background and returns immediately."""
    cmd = " ".join(shlex.quote(c) for c in exe() + ["refresh", name, "--if-changed"])
    return f"( {cmd} >> {shlex.quote(str(HOME / f'{name}.refresh.log'))} 2>&1 & )"


# ---- trigger: git hooks (main only) ---------------------------------------------------------------------------------
def install_git_hooks(root: Path, name: str) -> list[str]:
    hooks = Path(git(root, "rev-parse", "--git-path", "hooks") or ".git/hooks")
    hooks = hooks if hooks.is_absolute() else root / hooks
    hooks.mkdir(parents=True, exist_ok=True)
    body = (f"{MARK_START}\n# {HOOK_TAG}: refresh the project memory after changes land on main\n"
            "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)\n"
            'if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then\n'
            f"  {detached(name)}\nfi\n{MARK_END}\n")
    done = []
    for hook in ("post-merge", "post-commit"):
        f = hooks / hook
        text = f.read_text() if f.exists() else "#!/bin/sh\n"
        text = _strip_block(text) + ("\n" if not text.endswith("\n") else "") + body
        f.write_text(text)
        f.chmod(0o755)
        done.append(str(f))
    return done


def _strip_block(text: str) -> str:
    if MARK_START not in text:
        return text
    a, rest = text.split(MARK_START, 1)
    return a.rstrip("\n") + "\n" + (rest.split(MARK_END, 1)[1].lstrip("\n") if MARK_END in rest else "")


def remove_git_hooks(root: Path) -> None:
    hooks = Path(git(root, "rev-parse", "--git-path", "hooks") or ".git/hooks")
    hooks = hooks if hooks.is_absolute() else root / hooks
    for hook in ("post-merge", "post-commit"):
        f = hooks / hook
        if f.exists() and MARK_START in f.read_text():
            f.write_text(_strip_block(f.read_text()))


# ---- trigger: every N hours -----------------------------------------------------------------------------------------
def install_schedule(name: str, hours: float) -> str:
    log = str(HOME / f"{name}.refresh.log")
    if platform.system() == "Darwin":
        label = f"ai.defrost.{name}"
        plist = Path.home() / f"Library/LaunchAgents/{label}.plist"
        args = "".join(f"<string>{a}</string>" for a in exe() + ["refresh", name, "--if-changed"])
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{label}</string>
<key>ProgramArguments</key><array>{args}</array>
<key>StartInterval</key><integer>{int(hours * 3600)}</integer>
<key>StandardOutPath</key><string>{log}</string><key>StandardErrorPath</key><string>{log}</string>
<key>RunAtLoad</key><false/>
</dict></plist>
""")
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        subprocess.run(["launchctl", "load", str(plist)], capture_output=True)
        return str(plist)
    cmd = " ".join(shlex.quote(c) for c in exe() + ["refresh", name, "--if-changed"])
    line = f"0 */{max(1, int(hours))} * * * {cmd} >> {shlex.quote(log)} 2>&1  # {HOOK_TAG} {name}"
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    keep = [l for l in cur.splitlines() if f"{HOOK_TAG} {name}" not in l]
    subprocess.run(["crontab", "-"], input="\n".join(keep + [line]) + "\n", text=True)
    return "crontab"


def remove_schedule(name: str) -> None:
    if platform.system() == "Darwin":
        plist = Path.home() / f"Library/LaunchAgents/ai.defrost.{name}.plist"
        if plist.exists():
            subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
            plist.unlink()
        return
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if f"{HOOK_TAG} {name}" in cur:
        keep = [l for l in cur.splitlines() if f"{HOOK_TAG} {name}" not in l]
        subprocess.run(["crontab", "-"], input="\n".join(keep) + "\n", text=True)


# ---- trigger: Claude Code SessionStart hook -------------------------------------------------------------------------
def install_claude_hook(root: Path, name: str) -> str:
    f = root / ".claude/settings.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(f.read_text()) if f.exists() and f.read_text().strip() else {}
    hooks = settings.setdefault("hooks", {})
    starts = [h for h in hooks.get("SessionStart", []) if HOOK_TAG not in json.dumps(h)]
    starts.append({"hooks": [{"type": "command", "command": f"{detached(name)}  # {HOOK_TAG}"}]})
    hooks["SessionStart"] = starts
    f.write_text(json.dumps(settings, indent=2) + "\n")
    return str(f)


def remove_claude_hook(root: Path) -> None:
    f = root / ".claude/settings.json"
    if not f.exists():
        return
    settings = json.loads(f.read_text() or "{}")
    starts = [h for h in settings.get("hooks", {}).get("SessionStart", []) if HOOK_TAG not in json.dumps(h)]
    if "hooks" in settings:
        settings["hooks"]["SessionStart"] = starts
        if not starts:
            settings["hooks"].pop("SessionStart")
        if not settings["hooks"]:
            settings.pop("hooks")
    f.write_text(json.dumps(settings, indent=2) + "\n")


# ---- doc rules ------------------------------------------------------------------------------------------------------
def install_doc_rules(root: Path) -> str:
    from importlib import resources
    kit = resources.files("defrost_ai.assets") / "doc_rules"
    r = subprocess.run(["bash", str(kit / "install.sh"), str(root)], capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip()


# ---- memory rule in CLAUDE.md (agents skip MCP tools that CLAUDE.md does not mention) --------------------------------
RULE_START, RULE_END = "<!-- defrost-ai:memory:start -->", "<!-- defrost-ai:memory:end -->"


def install_memory_rule(root: Path, name: str, doc_trust: str | None = None) -> str:
    from defrost_ai import trust
    level = doc_trust or trust.DEFAULT
    f = root / "CLAUDE.md"
    text = f.read_text() if f.exists() else ""
    block = f"""{RULE_START}
## Project memory (defrost-ai)
Tools (MCP server `defrost`): `search`, `docs_for`, `remember`, `refresh`. Memory `{name}`.
- **Answer:** for "how do I / why does / what happens when" questions, call `search` first and cite
  `path:Lstart-end`. Use grep for exact strings and code.
{trust.RULE[level]}
- **Doc/code conflicts are the user's call:** ask only about disagreements you verified in the code (a `! doc may be
  stale` line alone means read the code, not ask), at most 2 questions per answer, the rest batched into one list at
  the end. Show both sides (doc `path:L..`, code `path:line`), ask with
  AskUserQuestion ("Code is right: update the doc" / "Doc is right: the code is a bug" / "Not a conflict" / "Not
  sure"), save the answer with `remember(kind="decision")`, then act on it. Hits with `resolved:` are decided.
- **Keep docs current:** after changing files, call `docs_for` with them and update those sections in the same
  change; new commands, env vars and config keys get a section (rules: `docs/DOC_RULES.md`).
- **Long tasks:** before /clear, save a handoff note with `remember(kind="note")`.
{RULE_END}"""
    text = re.sub(rf"\n*{re.escape(RULE_START)}.*?{re.escape(RULE_END)}\n*", "\n", text, flags=re.S).rstrip()
    i = text.find("<!-- defrost-ai:doc-rules:start -->")              # keep the doc-rules block last
    text = (text[:i].rstrip() + "\n\n" + block + "\n\n" + text[i:] if i >= 0
            else text + ("\n\n" if text else "") + block)
    f.write_text(text.rstrip() + "\n")
    return str(f)


def remove_memory_rule(root: Path) -> None:
    f = root / "CLAUDE.md"
    if f.exists():
        text = re.sub(rf"\n*{re.escape(RULE_START)}.*?{re.escape(RULE_END)}\n*", "\n", f.read_text(), flags=re.S)
        f.write_text(text.rstrip() + "\n")


# ---- setup ----------------------------------------------------------------------------------------------------------
def setup(path=".", domain=None, build="now", on_main_merge=False, every_hours=None, claude_hook=False,
          doc_rules=False, claude=False, remove=False, log=print, doc_trust=None, handoff=False, docs_sync=False,
          docs_auto=False, docs_budget=0.5, handoff_on_compact=False, docs_auto_merge=False,
          memory_dir="defrost-memory", docs_gate=True) -> dict:
    """doc_trust: "high" | "low" | None (keep the stored level; "low" for a new domain). See defrost_ai/trust.py.
    memory_dir: folder in the project for the context repository (None keeps it in ~/.defrost-ai)."""
    root = Path(path).expanduser().resolve()
    name = domain or root.name.lower().replace(" ", "-")
    out = {"domain": name, "path": str(root)}
    if remove:
        remove_git_hooks(root); remove_schedule(name); remove_claude_hook(root); remove_memory_rule(root)
        from defrost_ai import compact_handoff, docsync, notes
        notes.remove_hook(root)
        compact_handoff.remove_hook(root)
        docsync.remove_hooks(root)
        state_file(name).unlink(missing_ok=True)
        log(f"[{name}] triggers removed")
        return out | {"removed": True}
    ws = workspace_for(root, name)
    from defrost_ai import context_repo
    if memory_dir:
        spec = json.loads(ws.read_text())
        for comp in spec["components"]:
            if Path(comp["path"]).expanduser().resolve() == root and memory_dir not in comp.setdefault("exclude", []):
                comp["exclude"].append(memory_dir)               # indexed as <name>-context, not twice
        ws.write_text(json.dumps(spec, indent=1))
    out["context_repo"] = context_repo.place(name, root if memory_dir else None, memory_dir or "defrost-memory")
    from defrost_ai import trust
    if doc_trust:
        trust.write(ws, doc_trust)
    out["doc_trust"] = level = trust.read(ws)
    from defrost_ai.library import register
    register(name, ws, f"{root.name} (docs + code)")
    if claude:
        from defrost_ai.integrations.claude import install
        out["claude"] = install(root)
    if doc_rules:
        out["doc_rules"] = install_doc_rules(root)
    out["memory_rule"] = install_memory_rule(root, name, level)
    trig = {}
    if on_main_merge:
        if git(root, "rev-parse", "--git-dir"):
            trig["on_main_merge"] = install_git_hooks(root, name)
        else:
            log(f"[{name}] not a git repository: on-main-merge skipped")
    if every_hours:
        trig["every_hours"] = {"hours": every_hours, "agent": install_schedule(name, every_hours)}
    if claude_hook:
        trig["claude_hook"] = install_claude_hook(root, name)
    if handoff:
        from defrost_ai import notes
        notes.register_notes(name)                                   # context repo + its search domain
        trig["handoff"] = notes.install_hook(root)
    if handoff_on_compact:
        from defrost_ai import compact_handoff
        trig["handoff_on_compact"] = compact_handoff.install_hook(root)
    if docs_sync:
        from defrost_ai import docsync
        trig["docs_sync"] = {"auto": docs_auto, "budget_usd": docs_budget if docs_auto else 0,
                             "auto_merge": bool(docs_auto and docs_auto_merge),
                             "gate": docs_gate,
                             "files": docsync.install_hooks(root, auto=docs_auto, budget_usd=docs_budget,
                                                            auto_merge=docs_auto_merge, gate=docs_gate)}
    state_file(name).write_text(json.dumps({"path": str(root), "installed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            "triggers": trig}, indent=1))
    out["triggers"] = trig
    if build == "now":
        log(f"[{name}] building the memory now (first build loads the models; minutes for a large repo)…")
        out["build"] = refresh(name, if_changed=False, log=log)
    elif build == "background":
        subprocess.Popen(["/bin/sh", "-c", detached(name)])
        out["build"] = "started in background"
    return out


def status(domain=None) -> list[dict]:
    from defrost_ai.library import read_registry
    rows = []
    for name, reg in read_registry()["domains"].items():
        if domain and name != domain:
            continue
        if not Path(reg["workspace"]).expanduser().exists():          # files deleted by hand: report, don't crash
            rows.append({"domain": name, "built_at": None, "counts": None, "stale": True,
                         "why": f"workspace file missing ({reg['workspace']}); run `defrost setup` in the repo again",
                         "triggers": []})
            continue
        stale, why = changed_since_build(name)
        st = json.loads(state_file(name).read_text()) if state_file(name).exists() else {}
        man_f = Path(json.loads(Path(reg["workspace"]).read_text())["out"]).expanduser() / "manifest.json"
        man = json.loads(man_f.read_text()) if man_f.exists() else {}
        rows.append({"domain": name, "built_at": man.get("built_at"), "counts": man.get("counts"),
                     "stale": stale, "why": why, "triggers": sorted(st.get("triggers", {}))})
    return rows
