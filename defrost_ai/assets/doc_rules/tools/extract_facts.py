#!/usr/bin/env python3
"""Extract Facts blocks (`- Subject → relation → Object (qualifier)`) from Markdown docs as JSONL triples.

    python docs/tools/extract_facts.py docs > facts.jsonl

Each row: {"subject", "relation", "object", "qualifier", "path", "line", "section"}. Backticks are kept in the
text and also reported as "subject_code"/"object_code" so the triples can be joined to a code graph."""
import json
import re
import sys
from pathlib import Path

FACT = re.compile(r"^- (.+?) (?:→|->) (.+?) (?:→|->) (.+?)(?: \((.*)\))?\s*$")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def code(s):
    m = re.fullmatch(r"`([^`]+)`", s.strip())
    return m.group(1) if m else None


def extract(path):
    section, in_fence, in_facts = None, False, False
    for n, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        h = HEADING.match(line)
        if h:
            section, in_facts = h.group(1).strip(), False
            continue
        if line.strip() == "Facts:":                      # only the bullet list right after "Facts:"
            in_facts = True
            continue
        if not line.startswith("- "):
            in_facts = False
            continue
        m = FACT.match(line.strip()) if in_facts else None
        if m:
            s, r, o, q = (x.strip() if x else None for x in m.groups())
            yield {"subject": s, "relation": r, "object": o, "qualifier": q, "subject_code": code(s),
                   "object_code": code(o), "path": str(path), "line": n, "section": section}


def main(args):
    files = []
    for a in args or ["docs"]:
        p = Path(a)
        files += [f for f in sorted(p.rglob("*.md")) if "templates" not in f.parts] if p.is_dir() else [p]
    for f in files:
        for row in extract(f):
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
