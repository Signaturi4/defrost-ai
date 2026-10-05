#!/usr/bin/env python3
"""Install the doc rules into a project (stdlib only; macOS, Linux and Windows).

    python install.py /path/to/project                 # rules block at the end of CLAUDE.md
    python install.py /path/to/project --agents-md     # also AGENTS.md (Codex, Cursor, other agents)
    python install.py /path/to/project --remove        # take the block out again (copied files stay)

Copies into <project>/docs:
    DOC_RULES.md                                     the full rules, read only when docs are written
    templates/PAGE.template.md, GLOSSARY.template.md starting points (templates/ is never linted or indexed)
    tools/doc_lint.py, tools/extract_facts.py        the linter and the Facts -> JSONL extractor
and puts the short rule block at the end of CLAUDE.md. Re-running replaces the block instead of duplicating it.
The block uses the same markers as defrost-ai's `defrost setup`, so either tool can update it."""
import argparse
import re
import shutil
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"\n*<!-- defrost-ai:doc-rules:start -->.*?<!-- defrost-ai:doc-rules:end -->\n*", re.S)


def place_block(md: Path, snippet: str | None) -> str:
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    had = bool(BLOCK.search(text))
    text = BLOCK.sub("\n", text).rstrip()
    if snippet:
        text = text + ("\n\n" if text else "") + snippet.strip()
    md.write_text(text + "\n" if text else "", encoding="utf-8", newline="\n")
    return ("replaced" if had else "added") if snippet else ("removed" if had else "not present")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", nargs="?", default=".")
    ap.add_argument("--agents-md", action="store_true", help="also place the block in AGENTS.md")
    ap.add_argument("--remove", action="store_true", help="remove the block from CLAUDE.md / AGENTS.md")
    a = ap.parse_args(argv)
    root = Path(a.project).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    targets = [root / "CLAUDE.md"] + ([root / "AGENTS.md"] if a.agents_md else [])
    if a.remove:
        for md in targets:
            print(f"{md.name}: block {place_block(md, None)}")
        return
    docs = root / "docs"
    copies = {
        SKILL / "references" / "DOC_RULES.md": docs / "DOC_RULES.md",
        SKILL / "assets" / "PAGE.template.md": docs / "templates" / "PAGE.template.md",
        SKILL / "assets" / "GLOSSARY.template.md": docs / "templates" / "GLOSSARY.template.md",
        SKILL / "scripts" / "doc_lint.py": docs / "tools" / "doc_lint.py",
        SKILL / "scripts" / "extract_facts.py": docs / "tools" / "extract_facts.py",
    }
    for src, dst in copies.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"copied {dst.relative_to(root).as_posix()}")
    snippet = (SKILL / "assets" / "CLAUDE.snippet.md").read_text(encoding="utf-8")
    for md in targets:
        print(f"{md.name}: rule block {place_block(md, snippet)} (at the end)")
    print("next: create docs/GLOSSARY.md from docs/templates/GLOSSARY.template.md when you add the first term")


if __name__ == "__main__":
    main()
