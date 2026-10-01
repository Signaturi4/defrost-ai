"""Documents -> sections. A section is the text under one heading (split at paragraph boundaries past MAX_WORDS),
kept verbatim with its heading path and 1-based line range so every answer can be cited.

Markdown is read directly; reStructuredText and AsciiDoc headings are rewritten to '#' headings line for line, so
line numbers still point into the original file."""
from __future__ import annotations

import re

MAX_WORDS = 300
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+.:_])\1{2,}\s*$")
ADOC_HEADING = re.compile(r"^(={1,6})\s+(\S.*)$")


def rst_to_markdown_headings(text: str) -> str:
    """RST titles (underlined, optionally overlined) -> ATX headings; levels follow the order styles first appear."""
    lines, out, styles, i = text.splitlines(), [], [], 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        over = RST_UNDERLINE.match(line) and i + 2 < len(lines) and RST_UNDERLINE.match(lines[i + 2]) and lines[i + 1].strip()
        if over:
            style = ("over", line.strip()[0])
            title, skip = lines[i + 1].strip(), 3
        elif line.strip() and not RST_UNDERLINE.match(line) and RST_UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 1:
            style = ("under", nxt.strip()[0])
            title, skip = line.strip(), 2
        else:
            out.append(line); i += 1; continue
        if style not in styles:
            styles.append(style)
        out.append("#" * min(6, styles.index(style) + 1) + " " + title)
        out += [""] * (skip - 1)                      # keep line numbers aligned with the source file
        i += skip
    return "\n".join(out)


def asciidoc_to_markdown_headings(text: str) -> str:
    """'== Title' -> '## Title'; '----' / '....' listing delimiters -> ``` fences (no heading splits inside code)."""
    out = []
    for line in text.splitlines():
        m = ADOC_HEADING.match(line)
        if m:
            out.append("#" * len(m.group(1)) + " " + m.group(2))
        elif re.fullmatch(r"(-{4,}|\.{4,})\s*", line):
            out.append("```")
        else:
            out.append(line)
    return "\n".join(out)


def split_sections(text: str, suffix: str = ".md", max_words: int = MAX_WORDS):
    """-> [(level, heading_path list, line_start, line_end, body)] with 1-based line numbers."""
    suffix = suffix.lower()
    if suffix == ".rst":
        text = rst_to_markdown_headings(text)
    elif suffix in (".asc", ".adoc"):
        text = asciidoc_to_markdown_headings(text)
    lines = text.splitlines()
    out, stack, cur, start, fenced = [], [], [], 1, False

    def flush(end):
        body = "\n".join(cur).strip()
        if cur and HEADING.match(cur[0]) and not "\n".join(cur[1:]).strip():
            return                                           # heading only: its children carry it in their path
        if body:
            out.append((len(stack), [h for _, h in stack], start, end, body))

    for i, line in enumerate(lines, 1):
        if FENCE.match(line):
            fenced = not fenced
        m = None if fenced else HEADING.match(line)
        if m:
            flush(i - 1)
            level = len(m.group(1))
            stack = [(l, h) for l, h in stack if l < level] + [(level, m.group(2).strip())]
            cur, start = [line], i
        else:
            cur.append(line)
    flush(len(lines))
    split = []
    for level, path, a, b, body in out:                      # long sections: split at blank lines outside fences
        if len(body.split()) <= max_words:
            split.append((level, path, a, b, body)); continue
        part, n, line_no, fenced, pa = [], 0, a, False, a
        for line in body.splitlines():
            if FENCE.match(line):
                fenced = not fenced
            part.append(line); n += len(line.split())
            if n >= max_words and not line.strip() and not fenced:
                split.append((level, path, pa, line_no, "\n".join(part).strip())); part, n, pa = [], 0, line_no + 1
            line_no += 1
        if "\n".join(part).strip():
            split.append((level, path, pa, b, "\n".join(part).strip()))
    return split


def document_id(component: str, display_path: str) -> str:
    return f"{component}_{re.sub(r'[^a-z0-9_]', '_', display_path.lower())}"
