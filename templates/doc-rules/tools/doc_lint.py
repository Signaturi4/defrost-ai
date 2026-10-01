#!/usr/bin/env python3
"""Lint Markdown docs against DOC_RULES.md (stdlib only).

    python docs/tools/doc_lint.py docs/**/*.md README.md      # exit 1 if any ERROR

ERROR   missing/invalid frontmatter type, malformed Facts line, empty section heading
WARN    section < 60 or > 400 words, section opens with a pronoun, sentence > 25 words,
        reference/explanation section without Facts, code-shaped word not in backticks"""
import re
import sys
from pathlib import Path

TYPES = {"tutorial", "how-to", "reference", "explanation"}
PRONOUN_OPEN = re.compile(r"^(it|this|these|they|those|here|as (mentioned|noted|described) above)\b", re.I)
FACT = re.compile(r"^- (.+?) (?:→|->) (.+?) (?:→|->) (.+?)(?: \((.*)\))?\s*$")
CODEWORD = re.compile(r"(?<![`\w/])([a-z]+_[a-z0-9_]+|[a-z]+[A-Z]\w+|[A-Z][a-z]+[A-Z]\w*|\w+\(\))(?![`\w])")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def strip_code(text):
    return re.sub(r"```.*?```", " ", text, flags=re.S)


def sections(body):
    out, head, buf, start = [], None, [], 1
    in_fence = False
    for n, line in enumerate(body.splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            if head is not None or buf:
                out.append((head, start, buf))
            head, buf, start = m.group(2).strip(), [], n
        else:
            buf.append(line)
    out.append((head, start, buf))
    return out


def lint(path):
    issues = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    page_type = None
    if not fm:
        issues.append(("ERROR", 1, "missing YAML frontmatter (type, entity, status, updated)"))
    else:
        m = re.search(r"^type:\s*([\w-]+)", fm.group(1), re.M)
        page_type = m.group(1) if m else None
        if page_type not in TYPES:
            issues.append(("ERROR", 1, f"frontmatter type must be one of {sorted(TYPES)}"))
    body = text[fm.end():] if fm else text
    offset = text[:fm.end()].count("\n") if fm else 0
    for head, line, buf in sections(body):
        if head is None and not "".join(buf).strip():
            continue
        ln = line + offset
        if head is not None and not head:
            issues.append(("ERROR", ln, "empty heading"))
        raw = "\n".join(buf)
        prose = strip_code(raw)
        facts, other, in_facts = [], [], False
        for l in prose.splitlines():                       # Facts = the bullet list right after a "Facts:" line
            if l.strip() == "Facts:":
                in_facts = True
                continue
            if in_facts and l.startswith("- "):
                facts.append(l)
                continue
            in_facts = False
            other.append(l)
        for f in facts:
            if not FACT.match(f.strip()):
                issues.append(("ERROR", ln, f"malformed Facts line: {f.strip()[:80]}"))
        prose = "\n".join(l for l in other if not l.lstrip().startswith(("- ", "* ", "|")) or len(l.split()) > 3)
        words = len(re.findall(r"\w+", prose))
        is_table_only = prose.strip().startswith("|")
        if head and words and not is_table_only and (words < 60 or words > 400):
            issues.append(("WARN", ln, f"section '{head[:40]}' has {words} words (target 60-400)"))
        first = next((l.strip() for l in prose.splitlines() if l.strip() and not l.startswith(("|", "-", ">", "!"))), "")
        if head and PRONOUN_OPEN.match(first):
            issues.append(("WARN", ln, f"section '{head[:40]}' opens with a pronoun: '{first[:40]}'"))
        for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", prose)):
            if len(s.split()) > 25 and not s.lstrip().startswith(("|", "-")):
                issues.append(("WARN", ln, f"sentence of {len(s.split())} words: '{s[:60]}…'"))
        if head and page_type in ("reference", "explanation") and words >= 60 and not facts:
            issues.append(("WARN", ln, f"section '{head[:40]}' has no Facts block"))
        plain = re.sub(r"`[^`]*`", " ", prose)
        plain = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", plain)
        for w in sorted(set(CODEWORD.findall(plain)))[:5]:
            issues.append(("WARN", ln, f"code-shaped word not in backticks: {w}"))
    return issues


def main(paths):
    n_err = 0
    for p in paths:
        for level, line, msg in lint(p):
            n_err += level == "ERROR"
            print(f"{p}:{line}: {level}: {msg}")
    print(f"{len(paths)} file(s), {n_err} error(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
