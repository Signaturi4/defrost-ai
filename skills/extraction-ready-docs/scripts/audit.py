#!/usr/bin/env python3
"""Scan every Markdown file in a repository against the rules and split the work in two (stdlib only).

    python audit.py <project>            # report
    python audit.py <project> --json     # machine-readable

"Fix in place" items are edits inside a file (frontmatter, lifecycle, Sources, sections, sentences, backticks, Facts,
index lines, references): an agent applies them right away. "Needs a yes" items change paths other people rely on
(renames, moves into docs/, merges of duplicate copies, folder splits): an agent lists them and waits for approval.
Raw sources (`lifecycle: immutable`) only ever get frontmatter fixes, never body edits."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_lint  # noqa: E402
import repo_lint  # noqa: E402

SKIP_DIRS = {".git", "node_modules", "vendor", "third_party", "dist", "build", ".venv", "venv", "__pycache__",
             "templates", "archive", "site-packages"}
SKIP_FILES = {"CLAUDE.md", "AGENTS.md", "CHANGELOG.md", "LICENSE.md", "DOC_RULES.md", "KNOWLEDGE_RULES.md",
              "CODES.md", "SECURITY.md", "CODE_OF_CONDUCT.md"}
NEEDS_YES = {"name", "duplicate-copies", "duplicate-entity", "superseded-outside-archive", "crowded", "shared-prefix",
             "outside-docs"}


def markdown_files(root: Path):
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel[:-1]) or p.name in SKIP_FILES:
            continue
        yield p


def audit(root, docs="docs", mode=None):
    root = Path(root).resolve()
    items = defaultdict(lambda: {"fix_in_place": [], "needs_yes": []})
    mode = mode or repo_lint.read_mode(root)
    for p in markdown_files(root):
        rel = p.relative_to(root).as_posix()
        landing = p.parent == root and p.name == "README.md"
        for level, line, msg in doc_lint.lint(p):
            if landing and (level == "WARN" or "frontmatter" in msg):
                continue                                   # a landing README may skip frontmatter and section sizes
            items[rel]["fix_in_place"].append(f"{level} L{line}: {msg}")
        why = repo_lint.name_problem(p, repo_lint.read_codes(root / docs))
        if why and not landing:
            items[rel]["needs_yes"].append(f"rename: {why}")
        if not landing and p.parent == root and p.name != "README.md" or \
                (root / docs) not in p.parents and p.parent != root and p.name != "README.md":
            items[rel]["needs_yes"].append(f"outside-docs: move under {docs}/ (one home per file)")
    for i in repo_lint.lint_repo(root, docs, mode):
        rel = i.path.as_posix()
        if rel.endswith(".md") and any(rel.endswith(n) for n in SKIP_FILES):
            continue
        if i.code == "name":
            continue                                       # already reported above
        bucket = "needs_yes" if i.code in NEEDS_YES else "fix_in_place"
        items[rel][bucket].append(f"{i.level} [{i.code}] {i.msg}")
    return mode, {k: v for k, v in sorted(items.items()) if v["fix_in_place"] or v["needs_yes"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit every Markdown file against the documentation rules.")
    ap.add_argument("project", nargs="?", default=".")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    mode, items = audit(a.project, a.docs)
    if a.json:
        print(json.dumps({"mode": mode, "files": items}, indent=1))
        return 0
    n_fix = sum(len(v["fix_in_place"]) for v in items.values())
    n_yes = sum(len(v["needs_yes"]) for v in items.values())
    print(f"audit: {len(items)} file(s) to update — {n_fix} fix-in-place item(s), {n_yes} needing a yes "
          f"(lifecycle mode: {mode})")
    for path, v in items.items():
        print(f"\n{path}")
        for x in v["fix_in_place"]:
            print(f"  fix   {x}")
        for x in v["needs_yes"]:
            print(f"  ASK   {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
