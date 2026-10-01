"""Context repository: the project's working memory as a small git repo of Markdown files.

Handoff notes and the user's doc/code conflict decisions live here, one file per item, and every write is one commit,
so the history is auditable (`defrost context log`), can be shared through a remote, and is indexed as its own
search domain `<domain>-context`. The layout follows Letta Code's memory filesystem ("memfs v2"):

    <project>/defrost-memory/   git repo, branch main (after `defrost setup`; else ~/.defrost-ai/<domain>.context/)
      MEMORY.md                 root map (no frontmatter): what is here and how to use it
      project.md                core file (root *.md): read on every brief; keep small
      notes/MEMORY.md           index of handoff notes (rebuilt from frontmatter)
      notes/<stamp>-<slug>.md   one handoff note; notes/archive/ holds old ones after `context defrag`
      decisions/MEMORY.md       index of doc/code conflict decisions
      decisions/<stamp>-<slug>.md
      .memfs.config.json        limits (depth 2, 20k characters per file, 65k core); changing it needs approval

Every file except MEMORY.md has frontmatter with exactly `name` and `description` (plus a protected `read_only`).
A pre-commit hook enforces this (defrost_ai/context_constraints.py); a post-commit hook re-indexes the search
domain and, when enabled, pushes to the configured remote.

Ported from Letta Code (Apache-2.0, https://github.com/letta-ai/letta-code, commit 3687ea51):
  src/agent/memory-format.ts      root MEMORY.md marker, core = root *.md, projected paths need folder indexes
  src/agent/memory-git-hooks.ts   pre-commit validator + post-commit remote mirror -> install_hooks
  src/agent/memory-scanner.ts     tree listing for the map                         -> render_root_index
  src/skills/builtin/*            "defragmentation" maintenance                      -> defrag (deterministic part)
Changes: Python; the repo is per project domain (Letta: per agent); content is session handoffs and conflict
decisions (Letta: everything the agent learns); indexes are generated from frontmatter, not written by the agent;
no Letta server, sync is a plain git remote. See NOTICE."""
from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

from defrost_ai import context_constraints as cc
from defrost_ai import worktree as wt

FOLDERS = {"notes": "Handoff notes: goal, state, decisions and next steps of a work session, newest first.",
           "decisions": "Doc/code conflicts decided by the user: what the doc said, what the code does, the verdict."}
HOOK_TAG = "defrost-ai context repo"
BRIEF_WORDS = 450


def home() -> Path:
    return Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser()


DEFAULT_DIRNAME = "defrost-memory"


def _pointer(domain: str) -> Path:
    return home() / f"{domain}.context.path"


def repo_dir(domain: str) -> Path:
    """The repo lives in the project (`place`) when a pointer file says so, else under ~/.defrost-ai."""
    f = _pointer(domain)
    if f.exists() and f.read_text().strip():
        return Path(f.read_text().strip()).expanduser()
    return home() / f"{domain}.context"


def place(domain: str, project: str | Path | None, dirname: str = DEFAULT_DIRNAME) -> dict:
    """Put the repo where the user can see it: `<project>/<dirname>/` (project=None: back to ~/.defrost-ai).
    An existing repo is moved, under its operation lock. The folder is a separate git repo, so it is hidden from the
    project's git through .git/info/exclude (local, nothing tracked changes) and from the project's search domain."""
    old = repo_dir(domain)
    target = (Path(project).expanduser().resolve() / dirname) if project else home() / f"{domain}.context"
    out = {"domain": domain, "path": str(target), "moved": False}
    if old.resolve() != target.resolve():
        if target.exists() and any(target.iterdir()):
            raise ContextError(f"{target} exists and is not empty; pick another folder name")
        if (old / ".git").exists():
            with wt.operation_lock(old):
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.rmdir()
                shutil.move(str(old), str(target))
            wt.git(target, "worktree", "prune", check=False)
            install_hooks(target)
            out["moved"] = str(old)
    if project:
        _pointer(domain).parent.mkdir(parents=True, exist_ok=True)
        _pointer(domain).write_text(str(target) + "\n")
        out["git_exclude"] = _exclude_from_project(Path(project).expanduser().resolve(), dirname)
    else:
        _pointer(domain).unlink(missing_ok=True)
    if (home() / f"{context_domain(domain)}.workspace.json").exists() and exists(domain):
        register(domain)                                     # the search domain follows the move
    return out


def _exclude_from_project(project: Path, dirname: str) -> str | None:
    r = wt.git(project, "rev-parse", "--git-path", "info/exclude", check=False).strip()
    if not r:
        return None
    f = Path(r) if Path(r).is_absolute() else project / r
    f.parent.mkdir(parents=True, exist_ok=True)
    line = f"/{dirname}/"
    text = f.read_text() if f.exists() else ""
    if line not in text.splitlines():
        f.write_text(text + ("" if not text or text.endswith("\n") else "\n")
                     + f"# defrost-ai working memory (its own git repo)\n{line}\n")
    return str(f)


def context_domain(domain: str) -> str:
    return f"{domain}-context"


class ContextError(RuntimeError):
    pass


# ---- frontmatter -------------------------------------------------------------------------------------------------
def _scalar(text: str) -> str:
    return " ".join(str(text).split()).replace("---", "-")[:300] or "-"


def render(name: str, description: str, body: str) -> str:
    return f"---\nname: {_scalar(name)}\ndescription: {_scalar(description)}\n---\n\n{body.strip()}\n"


def parse(text: str) -> tuple[dict, str]:
    lines = text.split("\n")
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        return {}, text
    end = lines.index("---", 1)
    meta = {}
    for line in lines[1:end]:
        k, _, v = line.partition(":")
        if k.strip():
            meta[k.strip()] = v.strip()
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "item"


# ---- indexes (generated, so the agent never has to keep them in sync) ------------------------------------------
def _entries(folder: Path) -> list[tuple[str, dict]]:
    out = []
    for f in sorted(folder.glob("*.md"), reverse=True):                 # stamp-prefixed names: newest first
        if f.name != "MEMORY.md":
            out.append((f.name, parse(f.read_text(encoding="utf-8", errors="replace"))[0]))
    return out


def render_folder_index(folder: Path, about: str = "") -> str:
    rows = [f"- [{m.get('name', n)}]({n}): {m.get('description', '')}" for n, m in _entries(folder)]
    subs = sorted(d.name for d in folder.iterdir() if d.is_dir() and (d / "MEMORY.md").exists())
    text = f"# {folder.name}\n\n{about or FOLDERS.get(folder.name, '')}\n\n" + ("\n".join(rows) or "(empty)") + "\n"
    if subs:
        text += "\nSubfolders: " + ", ".join(f"[{s}/]({s}/MEMORY.md)" for s in subs) + "\n"
    return text


def render_root_index(root: Path, domain: str) -> str:
    core = [(f.name, parse(f.read_text(encoding="utf-8", errors="replace"))[0]) for f in sorted(root.glob("*.md"))
            if f.name != "MEMORY.md"]
    folders = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        n = len([f for f in d.glob("*.md") if f.name != "MEMORY.md"])
        folders.append(f"- [{d.name}/]({d.name}/MEMORY.md): {FOLDERS.get(d.name, '')} {n} file(s).")
    return (f"# Context repository: {domain}\n\n"
            "Working memory of this project: handoff notes and the user's doc/code decisions. Core files (below) are "
            "shown in every brief; open a folder's MEMORY.md for its index, then only the files you need. Search: "
            f"memory_search(query, domains=[\"{context_domain(domain)}\"]).\n\n"
            "## Core files\n" + ("\n".join(f"- [{m.get('name', n)}]({n}): {m.get('description', '')}" for n, m in core)
                                 or "(none)") +
            "\n\n## Folders\n" + ("\n".join(folders) or "(none)") + "\n")


def _rebuild_indexes(root: Path, domain: str, folders=None) -> list[str]:
    """Rewrite the folder indexes (all, or the given ones) and the root map; returns the changed paths."""
    changed = []
    targets = folders if folders is not None else [
        d.relative_to(root).as_posix() for d in root.rglob("*") if d.is_dir() and ".git" not in d.parts
        and not any(p.startswith(".") for p in d.relative_to(root).parts)]
    for rel in targets:
        d = root / rel
        if not d.is_dir():
            continue
        f = d / "MEMORY.md"
        text = render_folder_index(d)
        if not f.exists() or f.read_text() != text:
            f.write_text(text); changed.append(f"{rel}/MEMORY.md")
        parent = Path(rel).parent.as_posix()
        if parent != "." and parent not in targets:                       # a new subfolder must appear in its parent
            p = root / parent / "MEMORY.md"
            text = render_folder_index(root / parent)
            if not p.exists() or p.read_text() != text:
                p.write_text(text); changed.append(f"{parent}/MEMORY.md")
    f = root / "MEMORY.md"
    text = render_root_index(root, domain)
    if not f.exists() or f.read_text() != text:
        f.write_text(text); changed.append("MEMORY.md")
    return changed


# ---- repository --------------------------------------------------------------------------------------------------
def exists(domain: str) -> bool:
    return (repo_dir(domain) / ".git").exists()


def install_hooks(root: Path) -> list[str]:
    """pre-commit = the validator module's own source plus a main (self-contained, like Letta's hook);
    post-commit = re-index the search domain (if registered) and push (if `defrost.push` is true)."""
    hooks = wt.common_dir(root) / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    pre = hooks / "pre-commit"
    pre.write_text(f"#!{sys.executable}\n# {HOOK_TAG}: validate the staged memory tree\n"
                   + inspect.getsource(cc) + "\n\nif __name__ == \"__main__\":\n    sys.exit(hook_main(sys.argv[1:]))\n")
    exe = shutil.which("defrost") or f"{sys.executable} -m defrost_ai.cli"
    log = shlex.quote(str(home() / "context.log"))
    post = hooks / "post-commit"
    post.write_text(f"""#!/bin/sh
# {HOOK_TAG}: keep the search index fresh and mirror to the team remote (both opt-in via git config)
domain=$(git config --get defrost.domain)
if [ "$(git config --get defrost.reindex)" = "true" ] && [ -n "$domain" ]; then
  ( {exe} refresh "$domain-context" --if-changed >> {log} 2>&1 & )
fi
if [ "$(git config --get defrost.push)" = "true" ] && git remote get-url origin >/dev/null 2>&1; then
  ( git push -q origin HEAD >> {log} 2>&1 & )
fi
""")
    for f in (pre, post):
        f.chmod(0o755)
    return [str(pre), str(post)]


def init(domain: str, description: str = "") -> Path:
    """Create the repository (idempotent): layout, limits, hooks, first commit."""
    root = repo_dir(domain)
    if exists(domain):
        install_hooks(root)
        return root
    root.mkdir(parents=True, exist_ok=True)
    wt.git(root, "init", "-q", "-b", "main")
    wt.git(root, "config", "defrost.domain", domain)
    install_hooks(root)
    (root / cc.CONFIG_PATH).write_text(json.dumps(cc.DEFAULT_CONFIG, indent=2) + "\n")
    (root / "project.md").write_text(render(
        f"{domain} working memory",
        "Project-level facts every session should see; keep it short (core budget).",
        f"Domain `{domain}`. {description}".strip() + "\n\nAdd standing facts here (one line each): conventions the "
        "user asked for, decisions that apply everywhere. Details belong in notes/ or decisions/."))
    for folder in FOLDERS:
        (root / folder).mkdir(exist_ok=True)
    _rebuild_indexes(root, domain)
    wt.git(root, "add", "-A")
    wt.git(root, "commit", "-q", "-m", f"init: context repository for {domain}",
           env={cc.CONFIG_UPDATE_ENV: "1"})
    return root


def ensure(domain: str) -> Path:
    root = init(domain)
    migrate(domain)
    return root


def commit(domain: str, files: dict[str, str], message: str, delete=()) -> str:
    """Write `files` (repo-relative -> content), delete `delete`, rebuild the touched indexes, and commit only those
    paths (other edits in the checkout are left alone). The pre-commit hook validates; on rejection every touched
    file is restored and ContextError carries the hook's message."""
    root = ensure(domain) if not exists(domain) else repo_dir(domain)
    with wt.operation_lock(root):
        before = {p: ((root / p).read_text() if (root / p).exists() else None) for p in [*files, *delete]}
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        for rel in delete:
            (root / rel).unlink(missing_ok=True)
        folders = sorted({Path(p).parent.as_posix() for p in [*files, *delete] if Path(p).parent.as_posix() != "."})
        idx_before = {p: (root / p).read_text() if (root / p).exists() else None
                      for p in ["MEMORY.md", *[f"{d}/MEMORY.md" for d in folders]]}
        touched = sorted(set([*files, *delete, *_rebuild_indexes(root, domain, folders)]))
        wt.git(root, "add", "-A", "--", *touched)
        r = wt.subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", message, "--", *touched],
                              cwd=str(root), capture_output=True, text=True, env={**os.environ, **wt.AUTHOR})
        if r.returncode != 0:
            wt.git(root, "reset", "-q", "--", *touched, check=False)
            for p, text in {**before, **idx_before}.items():
                if text is None:
                    (root / p).unlink(missing_ok=True)
                else:
                    (root / p).write_text(text)
            raise ContextError((r.stderr or r.stdout).strip())
        return wt.git(root, "rev-parse", "--short", "HEAD").strip()


def log(domain: str, n: int = 20) -> list[dict]:
    if not exists(domain):
        return []
    out = wt.git(repo_dir(domain), "log", f"-{n}", "--format=%h\t%ad\t%an\t%s", "--date=format:%Y-%m-%d %H:%M")
    return [dict(zip(("sha", "date", "author", "subject"), line.split("\t", 3))) for line in out.splitlines()]


def files(domain: str, folder: str) -> list[Path]:
    d = repo_dir(domain) / folder
    return sorted((f for f in d.glob("*.md") if f.name != "MEMORY.md"), reverse=True) if d.exists() else []


def check(domain: str) -> list[str]:
    return cc.check_worktree(str(repo_dir(domain))) if exists(domain) else [f"no context repository for {domain}"]


# ---- progressive disclosure ------------------------------------------------------------------------------------
def map_brief(domain: str, max_words: int = BRIEF_WORDS) -> str:
    """Root map + core files, capped at `max_words` (never the folders' content: those are opened on demand)."""
    if not exists(domain):
        return ""
    root = repo_dir(domain)
    parts = [(root / "MEMORY.md").read_text()] if (root / "MEMORY.md").exists() else []
    for f in sorted(root.glob("*.md")):
        if f.name != "MEMORY.md":
            meta, body = parse(f.read_text())
            parts.append(f"## {meta.get('name', f.stem)} ({f.name})\n{body.strip()}")
    words = "\n\n".join(parts).split(" ")
    return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")


# ---- search domain -----------------------------------------------------------------------------------------------
def register(domain: str) -> Path:
    """Workspace + registry entry for `<domain>-context` (docs only; indexes and limits file excluded) and turn on
    re-indexing after each commit."""
    from defrost_ai.library import register as reg
    ensure(domain)
    ws = home() / f"{context_domain(domain)}.workspace.json"
    ws.write_text(json.dumps({"name": context_domain(domain), "out": str(home() / context_domain(domain)),
                              "components": [{"name": context_domain(domain), "path": str(repo_dir(domain)),
                                              "exclude": ["MEMORY.md", cc.CONFIG_PATH]}]}, indent=1))
    reg(context_domain(domain), ws, f"working memory of {domain}: handoff notes and doc/code decisions")
    wt.git(repo_dir(domain), "config", "defrost.reindex", "true")
    return ws


def set_remote(domain: str, url: str | None, push: bool = True) -> dict:
    """Team sharing: mirror every commit to `url` (off by default). url=None removes the remote and turns it off."""
    root = ensure(domain)
    wt.git(root, "remote", "remove", "origin", check=False)
    if url:
        wt.git(root, "remote", "add", "origin", url)
    wt.git(root, "config", "defrost.push", "true" if (url and push) else "false")
    return {"domain": domain, "remote": url, "push": bool(url and push)}


# ---- one-time migration from the pre-context-repo storage -------------------------------------------------------
def migrate(domain: str) -> int:
    """Old handoff notes (~/.defrost-ai/<domain>-notes/notes/*.md) and conflict decisions
    (~/.defrost-ai/<domain>.conflicts.jsonl) -> files in this repo, one commit. Old sources are renamed *.migrated."""
    old_notes = home() / f"{domain}-notes" / "notes"
    old_conf = home() / f"{domain}.conflicts.jsonl"
    new = {}
    if old_notes.is_dir():
        for f in sorted(old_notes.glob("*.md")):
            text = f.read_text()
            title = (re.search(r"^# (.+)$", text, re.M) or [None, f.stem])[1]
            goal = re.search(r"^## Goal.*?$\n(.+?)$", text, re.M)
            new[f"notes/{f.name}"] = render(title, (goal.group(1) if goal else title)[:200], text)
    if old_conf.exists():
        from defrost_ai import conflicts
        for row in (json.loads(line) for line in old_conf.read_text().splitlines() if line.strip()):
            rel, text = conflicts.render_decision(row)
            new[rel] = text
    if not new:
        return 0
    commit(domain, new, f"migrate: {len(new)} item(s) from the old notes folder / conflicts log")
    if old_notes.is_dir():
        old_notes.rename(old_notes.with_name("notes.migrated"))
    if old_conf.exists():
        old_conf.rename(old_conf.with_name(old_conf.name + ".migrated"))
    return len(new)


# ---- maintenance (deterministic "defragmentation"; runs as a worktree job) ---------------------------------------
def _split(path: Path, limit: int) -> list[Path]:
    """Split an oversize file at its `## ` headings into parts below `limit` characters."""
    meta, body = parse(path.read_text())
    chunks, cur = [], ""
    for block in re.split(r"(?m)^(?=## )", body):
        if cur and len(cur) + len(block) > limit - 400:
            chunks.append(cur); cur = ""
        cur += block
    chunks.append(cur)
    if len(chunks) == 1:
        return []
    out = []
    for i, chunk in enumerate(chunks, 1):
        p = path.with_name(f"{path.stem}-part{i}.md")
        p.write_text(render(f"{meta.get('name', path.stem)} (part {i}/{len(chunks)})",
                            meta.get("description", ""), chunk[:limit - 400]))
        out.append(p)
    path.unlink()
    return out


def defrag(domain: str, keep_notes: int = 20, merge: bool = True) -> dict:
    """Archive handoff notes beyond the newest `keep_notes`, split files above the size limit, rebuild every index.
    Runs in a worktree on a `defrost/defrag/...` branch and fast-forwards main when clean.
    (An LLM reflection pass, Letta's sleep-time step, would plug in here as an extra fn on the same job: opt-in and
    budget-capped; not implemented.)"""
    root = ensure(domain)
    stats = {"archived": 0, "split": 0}

    def job(d: Path):
        cfg = cc.DEFAULT_CONFIG | (cc.parse_config((d / cc.CONFIG_PATH).read_text())
                                   if (d / cc.CONFIG_PATH).exists() else {})
        notes = sorted((f for f in (d / "notes").glob("*.md") if f.name != "MEMORY.md"), reverse=True)
        if len(notes) > keep_notes:
            (d / "notes/archive").mkdir(exist_ok=True)
            for f in notes[keep_notes:]:
                f.rename(d / "notes/archive" / f.name); stats["archived"] += 1
        for f in [*d.glob("*.md"), *d.glob("*/*.md"), *d.glob("*/*/*.md")]:
            if f.name != "MEMORY.md" and len(f.read_text()) > cfg["maxFileCharacters"]:
                stats["split"] += bool(_split(f, cfg["maxFileCharacters"]))
        _rebuild_indexes(d, domain)
    res = wt.run(root, "defrag", job, merge=merge, message=lambda: (
        f"defrag: archived {stats['archived']} note(s), split {stats['split']} file(s), rebuilt indexes"))
    return res | stats
