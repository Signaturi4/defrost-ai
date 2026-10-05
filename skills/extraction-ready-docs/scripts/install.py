#!/usr/bin/env python3
"""Install the documentation rules into a project (stdlib only; macOS, Linux and Windows).

    python install.py <project>              # set up (any repository)
    python install.py <project> --defrost    # defrost-ai projects: + linters and the lint rule
    python install.py <project> --remove     # take the rule block out again (copied files stay)
    optional: --lifecycle living | versioned (default per-document: the agent picks per file)

Set-up copies `DOC_RULES.md` and `KNOWLEDGE_RULES.md` to `docs/`, the templates to `docs/templates/`, creates the
`docs/README.md` map and `docs/decision-register.md` when they are missing (existing files are never overwritten),
and puts a short rule block at the end of `CLAUDE.md` — and of `AGENTS.md` too when the project has one. Defrost mode is for projects indexed by defrost-ai: it also copies the linters and the Facts extractor
to `docs/tools/`, adds a "lint before you finish" rule, and uses defrost-ai's block markers so `defrost setup` and
this installer update the same block. Re-running replaces the block instead of duplicating it."""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SKILL = Path(__file__).resolve().parent.parent
MARKERS = {"general": "extraction-ready-docs", "defrost": "defrost-ai:doc-rules"}
ANY_BLOCK = re.compile(r"\n*<!-- (?:extraction-ready-docs|defrost-ai:doc-rules):start -->.*?"
                       r"<!-- (?:extraction-ready-docs|defrost-ai:doc-rules):end -->\n*", re.S)
MODE_RULE = {
    "per-document": "choose `lifecycle: living | versioned | immutable` for each page with the table in "
                    "`docs/KNOWLEDGE_RULES.md` §3 and write it into the frontmatter.",
    "living": "edit files in place and write why in the commit message; git keeps the history, so no copies and "
              "no archive.",
    "versioned": "before a material change, copy the file to `archive/` as `<name>--superseded-YYYY-MM-DD.md` "
                 "(`status: superseded`); the current file keeps the clean name. Skip `archive/` when searching.",
}
CHECK = ("9. **Check:** run `python docs/tools/doc_lint.py <changed files>` and `python docs/tools/repo_lint.py`, "
         "and fix every ERROR before finishing.")


def block(mode: str, defrost: bool) -> str:
    body = (SKILL / "assets" / "CLAUDE.snippet.md").read_text(encoding="utf-8")
    body = body.replace("{{MODE}}", mode).replace("{{MODE_RULE}}", MODE_RULE[mode])
    body = body.replace("{{CHECK}}", CHECK if defrost else "").rstrip()
    tag = MARKERS["defrost" if defrost else "general"]
    return f"<!-- {tag}:start -->\n{body}\n<!-- {tag}:end -->"


def place_block(md: Path, snippet: str | None) -> str:
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    had = bool(ANY_BLOCK.search(text))
    text = ANY_BLOCK.sub("\n", text).rstrip()
    if snippet:
        text = text + ("\n\n" if text else "") + snippet
    if text or md.exists():
        md.write_text(text + "\n" if text else "", encoding="utf-8", newline="\n")
    return ("replaced" if had else "added") if snippet else ("removed" if had else "not present")


def copy(src: Path, dst: Path, root: Path, overwrite=True):
    if dst.exists() and not overwrite:
        print(f"kept    {dst.relative_to(root).as_posix()} (exists)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"copied  {dst.relative_to(root).as_posix()}")



def utf8_stdout():
    """Windows consoles default to a legacy code page; `→` and other text must not crash the output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

def main(argv=None):
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Install the documentation rules into a project.")
    ap.add_argument("project", nargs="?", default=".")
    ap.add_argument("--lifecycle", choices=sorted(MODE_RULE), default="per-document",
                    help="how files change: per-document (default; the agent decides per page), living, versioned")
    ap.add_argument("--defrost", action="store_true", help="defrost-ai mode: linters, lint rule, defrost markers")
    ap.add_argument("--remove", action="store_true", help="remove the block from CLAUDE.md / AGENTS.md")
    a = ap.parse_args(argv)
    root = Path(a.project).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    targets = [root / "CLAUDE.md"] + ([root / "AGENTS.md"] if (root / "AGENTS.md").exists() else [])
    if a.remove:
        for md in targets:
            print(f"{md.name}: block {place_block(md, None)}")
        return
    docs, ref, assets = root / "docs", SKILL / "references", SKILL / "assets"
    copy(ref / "DOC_RULES.md", docs / "DOC_RULES.md", root)
    copy(ref / "KNOWLEDGE_RULES.md", docs / "KNOWLEDGE_RULES.md", root)
    for t in ("PAGE.template.md", "GLOSSARY.template.md", "README.index.template.md",
              "DECISION_REGISTER.template.md", "CODES.template.md", "ARCHIVE_BANNER.md"):
        copy(assets / t, docs / "templates" / t, root)
    copy(assets / "DOCS_MAP.template.md", docs / "README.md", root, overwrite=False)
    copy(assets / "DECISION_REGISTER.template.md", docs / "decision-register.md", root, overwrite=False)
    if a.defrost:
        for tool in ("doc_lint.py", "repo_lint.py", "extract_facts.py"):
            copy(SKILL / "scripts" / tool, docs / "tools" / tool, root)
    snippet = block(a.lifecycle, a.defrost)
    for md in targets:
        print(f"{md.name}: rule block {place_block(md, snippet)} "
              f"({'defrost' if a.defrost else 'general'} mode, lifecycle {a.lifecycle})")
    import audit                                                # scan every existing .md against the rules
    _, items = audit.audit(root, mode=a.lifecycle)
    n_fix = sum(len(v["fix_in_place"]) for v in items.values())
    n_yes = sum(len(v["needs_yes"]) for v in items.values())
    print(f"\nnext: {len(items)} existing Markdown file(s) to bring in line — {n_fix} fix(es) to apply in place, "
          f"{n_yes} rename/move/merge proposal(s) that need a yes.")
    if items:
        print(f"      full list: python {Path(__file__).with_name('audit.py')} {root}")


if __name__ == "__main__":
    main()
