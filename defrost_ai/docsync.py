"""Keep docs in step with code: from a diff, find the doc sections to update and the changes that have no doc yet.

The plan is model-free (SQLite + the stored code graph + git), so hooks can run it on every commit or stop:

    defrost docs-plan                 # uncommitted changes vs HEAD (what the Stop hook checks)
    defrost docs-plan --staged        # what `git commit` is about to record (PreToolUse hook on git commit)
    defrost docs-plan --commit HEAD   # one commit (git post-commit hook)

For every changed code or config file it reports:
- update: doc sections linked to a changed symbol (the symbol's span overlaps a changed hunk), then sections linked to
  the file only; each with "stale since" when the file's last commit is newer than the doc's;
- undocumented: changed files, and new symbols, that no doc section links to;
- covered: doc files already edited in the same diff (their sections are not asked again).
Writing the docs is the agent's job (/document-changes), following docs/DOC_RULES.md."""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from defrost_ai import notes
from defrost_ai.config import CODE_SUFFIXES, DOC_SUFFIXES

SKIP = re.compile(r"(^|/)(tests?|__tests__|spec|fixtures|migrations|vendor|dist|build|node_modules)/|"
                  r"\.(test|spec)\.\w+$|(^|/)(package-lock\.json|uv\.lock|poetry\.lock|yarn\.lock|pnpm-lock\.yaml)$")
CONFIG = re.compile(r"(^|/)(Dockerfile[^/]*|docker-compose[^/]*\.ya?ml|compose[^/]*\.ya?ml|[^/]*crontab|Makefile|"
                    r"\.github/workflows/[^/]+\.ya?ml|[^/]+\.sh|[^/]+\.toml|deploy/[^/]+\.ya?ml|config/[^/]+\.ya?ml)$")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def git(root: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def diff_args(root: Path, staged: bool = False, commit: str | None = None) -> list[str]:
    if commit:
        parent = f"{commit}^" if git(root, "rev-parse", "--verify", "-q", f"{commit}^") else \
            "4b825dc642cb6eb9a060e54bf8d69288fbee4904"                      # empty tree: first commit
        return [parent, commit]
    return ["--cached"] if staged else ["HEAD"]


def changes(root: Path, staged: bool = False, commit: str | None = None) -> dict[str, dict]:
    """{path: {"status": A|M|D|R, "lines": [(start, end)]}} relative to the repo root."""
    args = diff_args(root, staged, commit)
    out = {}
    for line in git(root, "diff", "--name-status", "-M", *args).splitlines():
        parts = line.split("\t")
        status, path = parts[0][0], parts[-1]
        out[path] = {"status": status, "lines": []}
    if not commit and not staged:                                           # new files not yet added to git
        for path in git(root, "ls-files", "--others", "--exclude-standard").splitlines():
            out.setdefault(path, {"status": "A", "lines": []})
    for path, d in out.items():
        if d["status"] in "MR":
            hunks = git(root, "diff", "-U0", *args, "--", path)
            d["lines"] = [(int(a), int(a) + max(int(n or 1), 1) - 1) for a, n in HUNK.findall(hunks)]
            d["terms"] = changed_terms(hunks)
            d["added_names"] = added_names(hunks)
        elif d["status"] == "A" and is_code(path) and (root / path).exists():
            d["added_names"] = added_names("\n".join("+" + l for l in (root / path).read_text(errors="ignore")
                                                      .splitlines()))
    return out


NAME = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]{2,}|[A-Z][A-Z0-9]*_[A-Z0-9_]{2,}|[a-z][\w-]*:[a-z][\w:-]+)(?![\w-])")


IMPORT = re.compile(r"\s*(import\b|from\s+\S+\s+import\b|.*\brequire\()")
ENV = re.compile(r"(process\.env\.|import\.meta\.env\.|os\.environ(\.get)?\(?\[?[\"']|getenv\([\"']|ENV\[[\"']|"
                 r"\$\{?|^\s*(export\s+)?|^\s*-?\s*)\0", re.M)       # \0 marks the candidate name


def added_names(diff: str) -> set[str]:
    """Operator-facing names on added lines: env vars (`MAX_RETRIES`), CLI flags (`--dry-run`), script tasks
    (`access-code:ensure`). A new one that no doc mentions is a doc gap even if the file is documented."""
    def names(sign):
        lines = [l[1:] for l in diff.splitlines() if l.startswith(sign) and not l.startswith(sign * 3)]
        lines = [l for l in lines if not IMPORT.match(l)]                  # module specifiers are not doc material
        body = re.sub(r"https?://\S+", " ", "\n".join(lines))
        out = set()
        for m in NAME.findall(body):
            if m.startswith("node:") or (m[0].isupper() and not ENV.search(body.replace(m, "\0"))):
                continue                                     # UPPER_SNAKE only when read as an env var / config key
            out.add(m)
        return out
    return names("+") - names("-")                     # a rewritten line keeps its old names: those are not new


def changed_terms(diff: str) -> set[str]:
    """Distinctive identifiers on added/removed lines (env vars, commands, keys, names) — what a doc would name."""
    body = "\n".join(l[1:] for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))
    words = {w.strip(".:-") for w in re.findall(r"[A-Za-z_][\w:.-]{3,}", body)}
    return {w for w in words
            if len(w) >= 5 and ("_" in w or "-" in w or ":" in w or re.search(r"[a-z][A-Z]", w) or w.isupper())}


def is_doc(path: str) -> bool:
    return Path(path).suffix.lower() in DOC_SUFFIXES


def is_code(path: str) -> bool:
    return not SKIP.search(path) and (Path(path).suffix.lower() in CODE_SUFFIXES or bool(CONFIG.search(path)))


class DomainIndex:
    """The stored memory of one domain, read without models: sections, doc->code links, code symbols per file."""

    def __init__(self, domain: str):
        from defrost_ai import store
        from defrost_ai.library import read_registry
        ws = json.loads(Path(read_registry()["domains"][domain]["workspace"]).read_text())
        out = Path(ws["out"]).expanduser()
        self.domain, self.components = domain, ws["components"]
        self.db = store.open_readonly(out / store.KNOWLEDGE_DB)
        code = json.loads((out / store.CODE_GRAPH).read_text())
        self.by_file = defaultdict(list)                                    # repo-relative file -> [(line, node)]
        for n in code["nodes"]:
            rel = (n.get("source_file") or "").split("/", 1)[-1]
            loc = re.match(r"L(\d+)", n.get("source_location") or "")
            self.by_file[rel].append((int(loc.group(1)) if loc else 1, n))
        for v in self.by_file.values():
            v.sort(key=lambda x: x[0])
        self.links = defaultdict(set)                                       # node id -> section ids
        self.files_of = defaultdict(set)                                    # node id -> repo-relative files
        for rel, nodes in self.by_file.items():
            for _, n in nodes:
                self.files_of[n["id"]].add(rel)
        for sid, nid, mention in self.db.execute("SELECT section_id, node_id, mention FROM links"):
            self.links[nid].add((sid, mention or ""))

    def sections_for(self, node: dict, path: str) -> set[str]:
        """Sections linked to `node` whose mention really names it in `path`: a path-like mention must be a suffix
        of the path (graphify node ids collide for same-named files), and a bare lowercase word (`proxy`, `next`)
        is too ambiguous to trust."""
        out = set()
        for sid, m in self.links.get(node["id"], ()):
            m = m.strip("`").removeprefix("./")
            if "/" in m or re.search(r"\.\w{1,5}$", m):
                if not (path == m or path.endswith("/" + m)):
                    continue
            elif re.fullmatch(r"[a-z]+", m.rstrip("()")) and not m.endswith("()"):
                continue
            out.add(sid)
        return out

    def text(self, sid: str) -> str:
        r = self.db.execute("SELECT text FROM sections WHERE id=?", (sid,)).fetchone()
        return r[0] if r else ""

    def named_anywhere(self, name: str) -> bool:
        return self.db.execute("SELECT 1 FROM sections WHERE instr(text, ?) > 0 LIMIT 1", (name,)).fetchone() is not None

    def section(self, sid: str) -> dict:
        r = self.db.execute("SELECT path, line_start, line_end, heading_path FROM sections WHERE id=?", (sid,)).fetchone()
        return {"path": r[0].split("/", 1)[-1], "lines": [r[1], r[2]], "heading": r[3]} if r else {}

    def mentions(self, path: str) -> set[str]:
        """Sections that name the file in text (config files the AST graph does not model)."""
        tail = "/".join(Path(path).parts[-2:])                             # `dir/file`, never a bare common name
        q = "SELECT id FROM sections WHERE text LIKE ? OR text LIKE ?"
        return {r[0] for r in self.db.execute(q, (f"%{path}%", f"%`{tail}`%"))}

    def symbols(self, path: str, lines: list[tuple[int, int]]):
        """-> (changed symbol nodes, all nodes of the file). A symbol spans from its line to the next symbol's."""
        nodes = self.by_file.get(path, [])
        changed = []
        starts = sorted({a for a, _ in nodes})
        for start, n in nodes:
            later = [a for a in starts if a > start]
            end = later[0] - 1 if later else 10 ** 9
            if any(a <= end and b >= start for a, b in lines) and n.get("label") != Path(path).name:
                changed.append(n)
        return changed, [n for _, n in nodes]


def _last_commit_ts(root: Path, path: str) -> int:
    out = git(root, "log", "-1", "--format=%ct", "--", path).strip()
    return int(out) if out else 0


def plan(path: str | Path = ".", staged: bool = False, commit: str | None = None, domain: str | None = None) -> dict:
    root = notes.git_root(Path(path))
    name = domain or notes.domain_for(root)
    ch = changes(root, staged, commit)
    docs_edited = sorted(p for p in ch if is_doc(p))
    code = {p: d for p, d in ch.items() if is_code(p) and d["status"] != "D"}
    deleted = sorted(p for p, d in ch.items() if is_code(p) and d["status"] == "D")
    result = {"root": str(root), "domain": name, "source": commit or ("staged" if staged else "working tree"),
              "changed_code": sorted(code), "deleted_code": deleted, "covered_docs": docs_edited,
              "update": [], "undocumented": []}
    if not name:
        result["error"] = "no memory domain for this repository (run defrost setup . first)"
        return result
    idx = DomainIndex(name)
    want = defaultdict(lambda: {"reasons": set(), "files": set()})
    for p, d in list(code.items()) + [(p, {"status": "D", "lines": []}) for p in deleted]:
        changed_syms, file_nodes = idx.symbols(p, d["lines"])
        sym_secs = {n["label"]: idx.sections_for(n, p) for n in changed_syms}
        by_symbol = set().union(*sym_secs.values()) if sym_secs else set()
        by_file = {sid for n in file_nodes for sid in idx.sections_for(n, p)} | idx.mentions(p)
        for sid in by_symbol:
            want[sid]["reasons"].add("symbol changed: " + ", ".join(sorted(l for l, v in sym_secs.items() if sid in v)))
            want[sid]["files"].add(p)
        for sid in by_file - by_symbol:
            hit = sorted(t for t in d.get("terms", ()) if t in idx.text(sid))[:4]
            want[sid]["reasons"].add("deleted file" if d["status"] == "D" else
                                     f"names changed text: {', '.join(hit)}" if hit else "file changed")
            want[sid]["files"].add(p)
        if not by_file and d["status"] != "D":
            result["undocumented"].append({"file": p, "status": d["status"],
                                           "symbols": [n["label"] for n in (changed_syms or file_nodes)
                                                       if n.get("label") != Path(p).name][:8]})
        elif d["status"] == "A" or changed_syms:
            new = [n["label"] for n in changed_syms if not idx.sections_for(n, p)]
            if new and d["status"] == "A":
                result["undocumented"].append({"file": p, "status": "A", "symbols": new[:8]})
    result["new_names"] = []
    for p, d in code.items():
        missing = sorted(n for n in d.get("added_names", ()) if not idx.named_anywhere(n))
        if missing:
            result["new_names"].append({"file": p, "names": missing[:10]})
    for sid, w in want.items():
        s = idx.section(sid)
        if not s or s["path"] in docs_edited:
            continue
        doc_ts = _last_commit_ts(root, s["path"])
        code_ts = max(_last_commit_ts(root, f) for f in w["files"])
        result["update"].append(s | {"files": sorted(w["files"]), "reasons": sorted(w["reasons"]),
                                     "stale": bool(code_ts and doc_ts and code_ts > doc_ts) or commit is None,
                                     "priority": 0 if any(r.startswith(("symbol", "deleted", "names")) for r in
                                                          w["reasons"]) else 1})
    result["update"].sort(key=lambda u: (u["priority"], u["path"], u["lines"][0]))
    return result


def render(p: dict, max_items: int = 15) -> str:
    """Short task text for the agent (hook output / slash command)."""
    if p.get("error"):
        return p["error"]
    if not (p["update"] or p["undocumented"] or p.get("new_names")):
        return ""
    lines = [f"Documentation follow-up for {p['source']} in {p['domain']} "
             f"({len(p['changed_code'])} code/config files changed):"]
    first = [u for u in p["update"] if u["priority"] == 0]
    other = [u for u in p["update"] if u["priority"] > 0]
    if first:
        lines.append("Update these doc sections (they name the changed symbols or text):")
        for u in first[:max_items]:
            lines.append(f"- {u['path']}:L{u['lines'][0]}-{u['lines'][1]}  {u['heading'].split(' > ', 1)[-1]}  "
                         f"<- {', '.join(u['files'])} ({'; '.join(u['reasons'])})")
        if len(first) > max_items:
            lines.append(f"- … {len(first) - max_items} more (defrost docs-plan --json)")
    if other:
        by_file = defaultdict(list)
        for u in other:
            by_file[", ".join(u["files"])].append(f"{u['path']}:L{u['lines'][0]}")
        lines.append("Also linked to the changed files (check only if the change affects them):")
        for f, secs in by_file.items():
            lines.append(f"- {f}: {len(secs)} sections, e.g. {', '.join(secs[:3])}")
    if p["undocumented"]:
        lines.append("Changed code with no doc section yet (add one if it is user- or operator-facing):")
        for u in p["undocumented"][:max_items]:
            syms = f" ({', '.join(u['symbols'])})" if u["symbols"] else ""
            lines.append(f"- {u['file']}{' [new]' if u['status'] == 'A' else ''}{syms}")
    if p.get("new_names"):
        lines.append("New names that no doc mentions (env vars, flags, tasks; document them where the file is "
                     "documented, e.g. in a settings table):")
        for u in p["new_names"][:max_items]:
            lines.append(f"- {u['file']}: {', '.join(f'`{n}`' for n in u['names'])}")
    if p["covered_docs"]:
        lines.append("Already edited in this change: " + ", ".join(p["covered_docs"]))
    return "\n".join(lines)


def fingerprint(p: dict) -> str:
    import hashlib
    key = json.dumps([p["source"], p["changed_code"], [(u["path"], u["lines"]) for u in p["update"]],
                      p["undocumented"], p.get("new_names")], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


# ---- triggers ----------------------------------------------------------------------------------------------------
# Inside Claude Code (project .claude/settings.json, opt-in via `defrost setup . --docs-sync`):
#   Stop        -> after the agent finishes editing: block the stop once per change set and ask it to document
#   PreToolUse  -> `git commit` run by the agent: deny once per staged change set until docs are updated and staged
# Outside Claude (git post-commit): write a pending doc task for the commit; the SessionStart hook shows pending
# tasks, and with `--docs-auto` a budget-capped `claude -p "/document-changes <sha>"` writes them in the background,
# in a git worktree on branch defrost/docs/<sha> (defrost_ai/worktree.py): the user's checkout is never touched; the
# branch waits for review (`defrost context branches --repo .`) or is fast-forwarded with --docs-auto-merge.
DOCS_TAG = "defrost-ai:docsync"
MARK_START, MARK_END = "# >>> defrost-ai docsync >>>", "# <<< defrost-ai docsync <<<"
COMMIT = re.compile(r"(^|[;&|]\s*|\s)git\s+(-C\s+\S+\s+)?commit\b")
INSTRUCTION = ("Follow the /document-changes procedure: update those sections in place and add sections for new "
               "user- or operator-facing code, following docs/DOC_RULES.md (one topic per section, exact backticked "
               "names, Facts lines), then run `python docs/tools/doc_lint.py <changed docs>`. If a listed section is "
               "not affected, leave it. Then call memory_handoff with what you documented.")


def _state(domain: str, kind: str) -> Path:
    return notes.home() / f"{domain}.docs-{kind}"


def _acked(domain: str, key: str) -> bool:
    f = _state(domain, "ack.json")
    seen = json.loads(f.read_text()) if f.exists() else []
    if key in seen:
        return True
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps((seen + [key])[-200:]))
    return False


def _gate_key(p: dict) -> str:
    """Ask once per set of changed code files (not per doc edit), so the agent is never asked twice for one change."""
    import hashlib
    return hashlib.sha1(json.dumps([p["source"] != "staged", p["changed_code"]]).encode()).hexdigest()[:12]


def _actionable(p: dict) -> bool:
    return bool([u for u in p["update"] if u["priority"] == 0] or p["undocumented"] or p.get("new_names"))


def hook(event: str, payload: dict) -> dict | None:
    """Claude Code hook handler -> JSON to print, or None (allow / nothing to do)."""
    cwd = Path(payload.get("cwd") or ".")
    if event == "stop":
        if payload.get("stop_hook_active"):
            return None
        p = plan(cwd)
        if p.get("error") or not _actionable(p) or _acked(p["domain"], _gate_key(p)):
            return None
        return {"decision": "block", "reason": render(p) + "\n\n" + INSTRUCTION}
    if event == "commit":
        if not COMMIT.search(payload.get("tool_input", {}).get("command", "")):
            return None
        p = plan(cwd, staged=True)
        if p.get("error") or not _actionable(p) or _acked(p["domain"], _gate_key(p)):
            return None
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": "Commit paused once for documentation.\n" + render(p) + "\n\n" + INSTRUCTION +
            " Stage the doc changes with the code, then run the same git commit again (it will not be paused twice)."}}
    return None


def record_commit(path: str | Path = ".", commit: str = "HEAD") -> Path | None:
    """git post-commit: save the doc plan of the commit as a pending task (model-free, < 1 s)."""
    root = notes.git_root(Path(path))
    sha = git(root, "rev-parse", "--short", commit).strip()
    p = plan(root, commit=sha)
    if p.get("error") or not _actionable(p):
        return None
    d = _state(p["domain"], "pending")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sha}.json"
    f.write_text(json.dumps(p | {"subject": git(root, "log", "-1", "--format=%s", sha).strip()}, indent=1))
    return f


def pending(domain: str) -> list[dict]:
    d = _state(domain, "pending")
    return [json.loads(f.read_text()) | {"file": str(f)} for f in sorted(d.glob("*.json"))] if d.exists() else []


def resolve(domain: str, sha: str | None = None) -> int:
    n = 0
    for t in pending(domain):
        if sha is None or t["source"].startswith(sha) or sha.startswith(t["source"]):
            Path(t["file"]).unlink(missing_ok=True); n += 1
    return n


def pending_brief(domain: str | None) -> str:
    """SessionStart (startup/resume) text: one line per commit that still needs docs."""
    tasks = pending(domain) if domain else []
    if not tasks:
        return ""
    rows = [f"- {t['source']} {t.get('subject', '')[:70]}: {sum(u['priority'] == 0 for u in t['update'])} sections "
            f"to update, {len(t['undocumented'])} files and {sum(len(n['names']) for n in t.get('new_names', []))} "
            f"names undocumented" for t in tasks[-5:]]
    return ("[defrost-ai] Commits with documentation follow-ups (from the post-commit hook):\n" + "\n".join(rows) +
            "\nWhen the user agrees, run /document-changes <sha> for each (oldest first).")


def auto_command(sha: str, budget_usd: float = 0.5) -> list[str]:
    """The background run used by --docs-auto: edits docs only, never commits, capped in dollars."""
    return ["claude", "-p", f"/document-changes {sha}", "--permission-mode", "acceptEdits",
            "--allowedTools", "Read,Grep,Glob,Edit,Write,Bash(python docs/tools/doc_lint.py:*),"
            "Bash(defrost docs-plan:*),Bash(git show:*),Bash(git diff:*),Bash(git log:*),"
            "mcp__defrost__memory_search,mcp__defrost__memory_handoff",
            "--max-budget-usd", str(budget_usd)]


def install_hooks(root: Path, auto: bool = False, budget_usd: float = 0.5, auto_merge: bool = False) -> list[str]:
    """Project .claude/settings.json (Stop + PreToolUse on Bash + SessionStart pending) and git post-commit."""
    import shlex
    import shutil
    exe = shutil.which("defrost") or "defrost"
    f = root / ".claude/settings.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(f.read_text()) if f.exists() and f.read_text().strip() else {}
    hooks = settings.setdefault("hooks", {})
    want = {"Stop": {"hooks": [{"type": "command", "command": f"{exe} docs-hook stop  # {DOCS_TAG}"}]},
            "PreToolUse": {"matcher": "Bash", "hooks": [{"type": "command",
                                                         "command": f"{exe} docs-hook commit  # {DOCS_TAG}"}]},
            "SessionStart": {"matcher": "startup|resume",
                             "hooks": [{"type": "command", "command": f"{exe} docs-pending  # {DOCS_TAG}"}]}}
    for event, entry in want.items():
        hooks[event] = [h for h in hooks.get(event, []) if DOCS_TAG not in json.dumps(h)] + [entry]
    f.write_text(json.dumps(settings, indent=2) + "\n")
    hdir = Path(git(root, "rev-parse", "--git-path", "hooks").strip() or ".git/hooks")
    hdir = hdir if hdir.is_absolute() else root / hdir
    hdir.mkdir(parents=True, exist_ok=True)
    log = shlex.quote(str(notes.home() / "docsync.log"))
    run = f"{exe} docs-record HEAD >> {log} 2>&1"
    if auto:                                       # not inside Claude Code: its own commits are gated by PreToolUse
        merge = " --merge" if auto_merge else ""             # default: leave defrost/docs/<sha> for review
        run += (f'\n  if [ -z "$CLAUDECODE" ]; then ( {exe} docs-auto HEAD --budget {budget_usd}{merge} >> {log} 2>&1 & ); fi')
    body = f"{MARK_START}\n# {DOCS_TAG}: record documentation follow-ups for this commit\n{{\n  {run}\n}}\n{MARK_END}\n"
    post = hdir / "post-commit"
    text = post.read_text() if post.exists() else "#!/bin/sh\n"
    text = _strip(text) + ("\n" if not text.endswith("\n") else "") + body
    post.write_text(text)
    post.chmod(0o755)
    return [str(f), str(post)]


def _strip(text: str) -> str:
    if MARK_START not in text:
        return text
    a, rest = text.split(MARK_START, 1)
    return a.rstrip("\n") + "\n" + (rest.split(MARK_END, 1)[1].lstrip("\n") if MARK_END in rest else "")


def remove_hooks(root: Path) -> None:
    f = root / ".claude/settings.json"
    if f.exists():
        settings = json.loads(f.read_text() or "{}")
        for event in list(settings.get("hooks", {})):
            settings["hooks"][event] = [h for h in settings["hooks"][event] if DOCS_TAG not in json.dumps(h)]
            if not settings["hooks"][event]:
                settings["hooks"].pop(event)
        if "hooks" in settings and not settings["hooks"]:
            settings.pop("hooks")
        f.write_text(json.dumps(settings, indent=2) + "\n")
    hdir = Path(git(root, "rev-parse", "--git-path", "hooks").strip() or ".git/hooks")
    post = (hdir if hdir.is_absolute() else root / hdir) / "post-commit"
    if post.exists() and MARK_START in post.read_text():
        post.write_text(_strip(post.read_text()))
