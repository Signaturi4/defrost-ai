#!/usr/bin/env python3
"""Lint the knowledge layer of a docs tree against KNOWLEDGE_RULES.md (stdlib only; macOS, Linux, Windows).

    python docs/tools/repo_lint.py                    # repo root = current folder, docs in docs/
    python docs/tools/repo_lint.py <repo> --docs docs --lifecycle per-document

doc_lint.py checks how a page is written; this checks where files live, what they are called, how many copies exist
and how they change. Exit code 1 if any ERROR.

ERROR  duplicate-entity            two current pages with the same frontmatter `entity`
       duplicate-copies            version copies of one file (`x_v2.md`, `x_v3_FINAL.md`, `x (2).md`)
       superseded-outside-archive  a `status: superseded` page outside `archive/`
       lifecycle-missing           per-document mode: a page without `lifecycle:`
WARN   name                        file name breaks the naming rule (case, spaces, `_`, versions, partial dates)
       duplicate-content           two files with identical bytes
       crowded / shared-prefix     a folder over --max-files, or >= --prefix-min files sharing a first word
       not-indexed                 a file or folder missing from its folder `README.md`
       no-text-twin                a binary (PDF, image, deck, spreadsheet) without a same-name `.md`
       broken-ref                  a backticked or linked path that does not exist (`(planned)` lines skipped)
       deprecated-no-replacement   `status: deprecated` without `replaced_by:`
       archive-in-living           living mode, but an `archive/` folder exists (git keeps history)
       immutable-edited            a `lifecycle: immutable` file changed after its first commit"""
import argparse
import hashlib
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MODES = ("living", "versioned", "per-document")
SKIP_DIRS = {"archive", "templates", "tools", "node_modules", "__pycache__"}
STANDARD = {"README.md", "CLAUDE.md", "AGENTS.md", "RULES.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
            "CODES.md",
            "LICENSE", "LICENSE.md", "NOTICE", "CODEOWNERS", "DOC_RULES.md", "KNOWLEDGE_RULES.md", "GLOSSARY.md",
            "Makefile", "Dockerfile", ".gitignore"}
RULE_PAGES = {"DOC_RULES.md", "KNOWLEDGE_RULES.md", "CODES.md"}          # quote bad examples on purpose
CODE_EXT = {".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".sh", ".ps1", ".bat", ".go", ".rs", ".java", ".kt",
            ".swift", ".rb", ".php", ".c", ".h", ".cpp", ".cs", ".json", ".yaml", ".yml", ".toml", ".lock", ".html",
            ".css", ".scss", ".ipynb", ".sql"}
BINARY_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pptx", ".ppt", ".key", ".xlsx", ".xls",
              ".docx", ".doc", ".numbers", ".pages", ".mp3", ".mp4", ".m4a", ".wav", ".mov"}
BAD_EXT = {".bak", ".old", ".orig", ".tmp", ".swp"}
BAD_WORDS = {"final", "copy", "new", "old", "backup", "fixed", "latest", "draft2", "use", "this"}
FULL_DATE = r"\d{4}-\d{2}-\d{2}"
FM = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.S)
BLOCK = re.compile(r"<!-- (?:extraction-ready-docs|defrost-ai:doc-rules):start -->(.*?)"
                   r"<!-- (?:extraction-ready-docs|defrost-ai:doc-rules):end -->", re.S)
TICK = re.compile(r"`([^`\s]+)`")
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


@dataclass
class Issue:
    level: str
    code: str
    path: Path
    msg: str
    line: int = 1

    def __str__(self):
        return f"{self.path.as_posix()}:{self.line}: {self.level}: [{self.code}] {self.msg}"


def frontmatter(path: Path) -> dict:
    if path.suffix != ".md":
        return {}
    m = FM.match(path.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep and not line.startswith((" ", "\t")):
            out[k.strip()] = v.split("#")[0].strip().strip("'\"")
    return out


def read_mode(root: Path) -> str:
    """The lifecycle mode written into the CLAUDE.md / AGENTS.md rule block; per-document when none is set."""
    for name in ("CLAUDE.md", "AGENTS.md"):
        p = root / name
        if p.exists():
            m = BLOCK.search(p.read_text(encoding="utf-8", errors="ignore"))
            mm = re.search(r"Lifecycle mode:\s*\**\s*([a-z-]+)", m.group(1)) if m else None
            if mm and mm.group(1) in MODES:
                return mm.group(1)
    return "per-document"


def walk(base: Path):
    """Files under base, skipping archive/, templates/, tools/ and hidden folders."""
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel[:-1]):
            continue
        if p.is_file() and not p.name.startswith("."):
            yield p


CODE = re.compile(r"^([A-Z][A-Z0-9]*(?:_[A-Z][A-Z0-9]*)*)_(\d+)(?=-|$)")


def read_codes(docs: Path) -> dict | None:
    """Optional code legend for large doc sets: docs/CODES.md, a table whose first column is the code (`P1`, `MR`)."""
    p = docs / "CODES.md"
    if not p.exists():
        return None
    codes = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"[A-Z][A-Z0-9]*", cells[0]):
            codes[cells[0]] = cells[1]
    return codes


def name_problem(p: Path, codes: dict | None = None):
    if p.name in STANDARD or p.suffix.lower() in CODE_EXT:
        return None
    m = CODE.match(p.name.split(".")[0]) if codes is not None else None
    if m:                                                        # coded name: P1_MR_INT_01-maya-2026-10-12.md
        unknown = [c for c in m.group(1).split("_") if c not in codes]
        if unknown:
            return f"code {', '.join(unknown)} is not in docs/CODES.md"
        rest = p.name[m.end():].lstrip("-")
        if not rest.split(".")[0]:
            return "code without words: add what-who after it (`P1_MR_INT_01-maya-2026-10-12.md`)"
        return name_problem(p.with_name(rest), None)
    if p.suffix.lower() in BAD_EXT or p.name.endswith("~"):
        return f"backup copy `{p.name}`: edit the file itself, git keeps history"
    stem = p.name.split(".")[0]
    if re.search(r"[A-Z]", stem):
        return "uppercase letters: use lowercase words joined by hyphens"
    if re.search(r"[\s&()\[\]{},'!]", stem):
        return "spaces or special characters: use lowercase words joined by hyphens"
    if "_" in stem:
        return "underscore: join words with hyphens"
    words = stem.split("-")
    if any(re.fullmatch(r"v\d+", w) for w in words) or any(w in BAD_WORDS for w in words):
        return "version word in the name (`v2`, `final`, `copy`, `new`…): keep one file, git keeps history"
    if re.search(r"(?<!\d)\d{2}-\d{2}-\d{4}(?!\d)", stem):
        return "date not in YYYY-MM-DD order"
    for m in re.finditer(r"(?<!\d)\d{4}-\d{2}(?:-\d{2})?(?!\d)", stem):
        if not re.fullmatch(FULL_DATE, m.group(0)):
            return f"partial date `{m.group(0)}`: write the full date YYYY-MM-DD"
        tail = stem[m.end():]
        if tail and tail not in ("-confidential",) and not tail.startswith("--superseded"):
            return "date in the middle: the date goes at the end (before `-confidential`)"
    return None


def nearest_readme(p: Path, docs: Path):
    d = p.parent
    while True:
        r = d / "README.md"
        if r.exists() and r != p:
            return r
        if d == docs or docs not in d.parents:
            return None
        d = d.parent


def git_commits(root: Path, p: Path) -> int | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "--format=%H", "--", p.relative_to(root).as_posix()],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return len(out.stdout.split()) if out.returncode == 0 else None


def ref_exists(token: str, src: Path, root: Path, docs: Path, names: set) -> bool:
    t = token.split("#")[0].rstrip(".,;:)")
    if not t or "/" not in t and t in names:
        return True
    rel = t.lstrip("./") if t.startswith("./") else t
    for base in (src.parent, docs, root, *src.parents):
        if (base / rel).exists():
            return True
    return False


def looks_like_path(t: str) -> bool:
    if re.search(r"[<>*{}$|=]|YYYY|://|^-|^@|\(\)", t) or t.startswith(("~", "#", ".")):
        return False
    return "/" in t and not t.startswith("/") or bool(re.search(r"\.[a-z][a-z0-9]{0,4}$", t))


def lint_repo(root, docs="docs", mode=None, max_files=12, prefix_min=3) -> list:
    root = Path(root).resolve()
    docs = (root / docs).resolve()
    mode = mode or read_mode(root)
    issues = []
    if not docs.is_dir():
        return issues
    files = list(walk(docs))
    fm = {p: frontmatter(p) for p in files}
    names = {p.name for p in root.rglob("*") if p.is_file()}

    codes = read_codes(docs)
    for p in files:                                                  # names
        why = name_problem(p, codes)
        if why:
            issues.append(Issue("WARN", "name", p.relative_to(root), why))

    current = defaultdict(list)                                      # one current page per entity
    for p, f in fm.items():
        status = f.get("status", "current").lower()
        if f.get("entity") and status not in ("deprecated", "superseded"):
            current[f["entity"].strip().lower()].append(p)
        if status == "superseded":
            issues.append(Issue("ERROR", "superseded-outside-archive", p.relative_to(root),
                                "superseded copies live under archive/, out of the search path"))
        if status == "deprecated" and not f.get("replaced_by"):
            issues.append(Issue("WARN", "deprecated-no-replacement", p.relative_to(root),
                                "set `replaced_by: <path>` so readers find the current page"))
        lc = f.get("lifecycle")
        if mode == "per-document" and f and not lc and p.name != "README.md" and p.name not in RULE_PAGES:
            issues.append(Issue("ERROR", "lifecycle-missing", p.relative_to(root),
                                "per-document mode: add `lifecycle: living | versioned | immutable`"))
        if lc == "immutable":
            n = git_commits(root, p)
            if n and n > 1:
                issues.append(Issue("WARN", "immutable-edited", p.relative_to(root),
                                    f"immutable source changed in {n} commits: put corrections in a separate note"))
    for ent, ps in current.items():
        if len(ps) > 1:
            for p in ps:
                others = ", ".join(q.relative_to(root).as_posix() for q in ps if q != p)
                issues.append(Issue("ERROR", "duplicate-entity", p.relative_to(root),
                                    f"entity '{ent}' also has a current page: {others}"))

    family = defaultdict(list)                                       # version copies: x_v2, x_v3_FINAL, x (2)
    for p in files:
        if p.name in STANDARD or p.suffix.lower() in CODE_EXT:
            continue
        words = re.split(r"[\s_\-().]+", p.name.rsplit(".", 1)[0].lower())
        core = [w for w in words if w and not re.fullmatch(r"v\d+|\d|final|copy|new|old|backup|fixed|latest|use|this", w)]
        if core and len(core) < len([w for w in words if w]):
            family[(p.parent, "-".join(core), p.suffix.lower())].append(p)
    for (d, core, ext), ps in family.items():
        clean = d / f"{core}{ext}"
        group = ps + ([clean] if clean.exists() and clean not in ps else [])
        if len(group) > 1:
            issues.append(Issue("ERROR", "duplicate-copies", d.relative_to(root),
                                f"{len(group)} copies of '{core}': " + ", ".join(sorted(q.name for q in group))
                                + " — merge into one file; git or archive/ keeps old versions"))

    by_hash = defaultdict(list)                                      # identical copies
    for p in files:
        data = p.read_bytes()
        if data.strip():
            by_hash[hashlib.sha256(data).hexdigest()].append(p)
    for ps in by_hash.values():
        if len(ps) > 1:
            issues.append(Issue("WARN", "duplicate-content", ps[1].relative_to(root),
                                "identical to " + ", ".join(q.relative_to(root).as_posix() for q in ps if q != ps[1])))

    folders = defaultdict(list)                                      # crowding
    for p in files:
        if p.name != "README.md":
            folders[p.parent].append(p)
    for d, ps in folders.items():
        rel = d.relative_to(root)
        if len(ps) > max_files:
            issues.append(Issue("WARN", "crowded", rel, f"{len(ps)} files (> {max_files}): split into sub-folders "
                                                        "named by topic (see KNOWLEDGE_RULES.md, Folder splits)"))
        groups = defaultdict(list)
        for p in ps:
            first = p.name.split(".")[0].split("-")[0]
            if "-" in p.name and first:
                groups[first].append(p)
        for word, g in groups.items():
            if len(g) >= prefix_min and len(g) < len(ps):
                issues.append(Issue("WARN", "shared-prefix", rel,
                                    f"{len(g)} files start with '{word}-': move them to a `{word}s/` sub-folder"))

    for p in files:                                                  # folder indexes
        if p.name == "README.md" and p.parent == docs:
            continue
        readme = nearest_readme(p, docs) if p.name != "README.md" else nearest_readme(p.parent, docs)
        if readme is None:
            continue
        text = readme.read_text(encoding="utf-8", errors="ignore")
        if p.name == "README.md":
            if f"{p.parent.name}/" not in text:
                issues.append(Issue("WARN", "not-indexed", p.parent.relative_to(root),
                                    f"folder not listed in {readme.relative_to(root).as_posix()}"))
        elif p.name not in text:
            issues.append(Issue("WARN", "not-indexed", p.relative_to(root),
                                f"add one line for it to {readme.relative_to(root).as_posix()}"))
    if not (docs / "README.md").exists():
        issues.append(Issue("WARN", "not-indexed", docs.relative_to(root), "no `README.md` map of the docs folder"))

    stems = {(p.parent, p.name.split(".")[0]) for p in files if p.suffix == ".md"}
    for p in files:                                                  # text twins
        if p.suffix.lower() in BINARY_EXT and (p.parent, p.name.split(".")[0]) not in stems:
            issues.append(Issue("WARN", "no-text-twin", p.relative_to(root),
                                f"add `{p.name.split('.')[0]}.md` describing its content; search cannot read it"))

    md = [p for p in files if p.suffix == ".md" and p.name not in RULE_PAGES]
    md += [p for p in root.glob("*.md") if p.name not in RULE_PAGES]
    for p in md:                                                     # references
        fence = False
        text = p.read_text(encoding="utf-8", errors="ignore")
        text = BLOCK.sub(lambda m: "\n" * m.group(0).count("\n"), text)   # the rule block quotes examples
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```"):
                fence = not fence
            if fence:
                continue
            planned = lambda m: line[m.end():].lstrip().startswith("(planned)")  # noqa: E731
            tokens = [m.group(1) for m in TICK.finditer(line) if looks_like_path(m.group(1)) and not planned(m)]
            tokens += [m.group(1) for m in LINK.finditer(line)
                       if "://" not in m.group(1) and not m.group(1).startswith("mailto:") and not planned(m)]
            for t in tokens:
                if not ref_exists(t, p, root, docs, names):
                    issues.append(Issue("WARN", "broken-ref", p.relative_to(root),
                                        f"`{t}` does not exist (mark future files with `(planned)`)", n))

    if mode == "living" and any((d / "archive").is_dir() for d in (root, docs)):
        issues.append(Issue("WARN", "archive-in-living", Path("archive"),
                            "living mode keeps history in git; move to versioned mode or remove archive/"))
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lint the knowledge layer of a docs tree (KNOWLEDGE_RULES.md).")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--docs", default="docs", help="docs folder, relative to root (default: docs)")
    ap.add_argument("--lifecycle", choices=MODES, help="override the mode in the CLAUDE.md block")
    ap.add_argument("--max-files", type=int, default=12, help="files per folder before it counts as crowded")
    ap.add_argument("--prefix-min", type=int, default=3, help="files sharing a first word before a split is proposed")
    a = ap.parse_args(argv)
    issues = lint_repo(a.root, a.docs, a.lifecycle, a.max_files, a.prefix_min)
    for i in sorted(issues, key=lambda i: (i.level != "ERROR", str(i.path), i.line)):
        print(i)
    n_err = sum(i.level == "ERROR" for i in issues)
    print(f"mode {a.lifecycle or read_mode(Path(a.root).resolve())}: {n_err} error(s), "
          f"{len(issues) - n_err} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
