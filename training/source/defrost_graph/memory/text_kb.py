"""Text KB: every markdown document of every component, stored verbatim as sections, plus doc->code links.

A section is the text under one heading (split further at paragraph boundaries past MAX_WORDS), kept with its heading
path ("ARCHITECTURE.md > Feed path > Serve Lambda") and line range, so an answer can quote and cite it. Stored in
SQLite with an FTS5 index (BM25) next to code_kb.json.

Links follow the plan's "exact symbol match first" rule. Mentions are backticked spans, file paths, and code-shaped
words (camelCase with an inner capital, snake_case, dotted Terraform refs). A mention links when it names exactly one
code node, preferring the document's own component; 2-3 candidates are kept as AMBIGUOUS; more is too generic to use.
A backticked code-shaped mention that resolves to nothing is recorded: it is either outside the scanned code (a CLI
command, a flag) or a symbol that no longer exists, and the rate of those is one input to the doc-trust weight.

    uv run python -m defrost_graph.memory.text_kb --workspace defrost_graph/memory/workspaces/client.json"""
import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from defrost_graph.memory.workspace import DOC_SUFFIXES, git_times, load

MAX_WORDS = 300
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
BACKTICK = re.compile(r"`([^`\n]{2,120})`")
PATHLIKE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|jsx|tf|swift|kt|go|ya?ml|json|sh|sql))\b")
CODEWORD = re.compile(r"\b([a-z]+[A-Z][A-Za-z0-9]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+|[a-z][a-z0-9]*_[a-z0-9_]{2,}"
                      r"|(?:var|module|data|aws_[a-z0-9_]+)\.[A-Za-z0-9_.-]+)\b")
CODE_SHAPED = re.compile(r"[()_./]|[a-z][A-Z]|^[A-Z][a-z]+[A-Z]")

SCHEMA = """
CREATE TABLE docs (id TEXT PRIMARY KEY, component TEXT, path TEXT, abspath TEXT, sha256 TEXT, time INTEGER,
                   words INTEGER, title TEXT);
CREATE TABLE sections (id TEXT PRIMARY KEY, doc_id TEXT, component TEXT, path TEXT, heading_path TEXT, level INTEGER,
                       ordinal INTEGER, line_start INTEGER, line_end INTEGER, text TEXT, words INTEGER,
                       n_mentions INTEGER, n_resolved INTEGER, n_unresolved INTEGER);
CREATE VIRTUAL TABLE sections_fts USING fts5(heading_path, text, content='sections', content_rowid='rowid',
                                             tokenize='porter unicode61');
CREATE TABLE links (section_id TEXT, node_id TEXT, mention TEXT, confidence TEXT, score REAL);
CREATE TABLE unresolved (section_id TEXT, mention TEXT);
"""


RST_UNDER = re.compile(r"^([=\-~^\"'`#*+.:_])\1{2,}\s*$")


def rst_to_atx(text):
    """reStructuredText section titles (a line underlined, optionally overlined, with a repeated punctuation char)
    -> ATX '#' headings. RST assigns levels by the order in which underline styles first appear."""
    lines, out, styles, i = text.splitlines(), [], [], 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        over = RST_UNDER.match(line) and i + 2 < len(lines) and RST_UNDER.match(lines[i + 2]) and lines[i + 1].strip()
        if over:
            style = ("over", line.strip()[0])
            title, skip = lines[i + 1].strip(), 3
        elif line.strip() and not RST_UNDER.match(line) and RST_UNDER.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 1:
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


ADOC_HEAD = re.compile(r"^(={1,6})\s+(\S.*)$")


def adoc_to_atx(text):
    """AsciiDoc -> markdown shape, line for line: '== Title' -> '## Title', '----' / '....' listing delimiters ->
    ``` fences (so headings inside code are not split on)."""
    out = []
    for line in text.splitlines():
        m = ADOC_HEAD.match(line)
        if m:
            out.append("#" * len(m.group(1)) + " " + m.group(2))
        elif re.fullmatch(r"(-{4,}|\.{4,})\s*", line):
            out.append("```")
        else:
            out.append(line)
    return "\n".join(out)


def sections_of(text, max_words=MAX_WORDS, rst=False, adoc=False):
    """markdown (or reStructuredText) -> [(level, heading_path list, line_start, line_end, text)], 1-based lines."""
    if rst:
        text = rst_to_atx(text)
    if adoc:
        text = adoc_to_atx(text)
    lines = text.splitlines()
    out, stack, cur, start, fenced = [], [], [], 1, False

    def flush(end):
        body = "\n".join(cur).strip()
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


def label_index(nodes):
    """name -> [node ids]. Keys: the label without call parens, the file name for file nodes, the dotted Terraform key."""
    idx = defaultdict(set)
    for n in nodes:
        lab = (n.get("label") or "").strip()
        if not lab or len(lab) > 80 or n.get("kind") == "rationale" or n.get("file_type") == "rationale":
            continue
        key = lab[:-2] if lab.endswith("()") else lab
        key = key.lstrip(".")
        if len(key) >= 3:
            idx[key].add(n["id"])
        src = n.get("source_file") or ""
        if src and lab == Path(src).name:                     # file node: also index its path tail
            parts = src.split("/")[1:]                        # drop the component prefix
            for k in range(1, min(4, len(parts)) + 1):
                idx["/".join(parts[-k:])].add(n["id"])
    return idx


def resolve(mention, idx, comp_of, component):
    m = mention.strip().strip("`'\"").rstrip(".,:;")
    m = m[:-2] if m.endswith("()") else m
    cands = idx.get(m) or idx.get(m.split("(")[0]) or idx.get(m.rsplit(".", 1)[-1] if "." in m and "/" not in m
                                                              and not m.startswith(("var.", "module.", "aws_")) else "")
    if not cands:
        return None, []
    own = [c for c in cands if comp_of[c] == component]
    pick = sorted(own or cands)
    return m, pick


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    args = ap.parse_args(argv)
    ws = load(args.workspace)
    out = ws["out"]
    code = json.loads((out / "code_kb.json").read_text())
    idx = label_index(code["nodes"])
    comp_of = {n["id"]: n["component"] for n in code["nodes"]}

    db_path = out / "text_kb.sqlite"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    stats = defaultdict(lambda: defaultdict(int))
    for comp in ws["components"]:
        files = [(r, p) for r, p in comp.walk(DOC_SUFFIXES)]
        times = {}
        for root in comp.roots:
            times.update(git_times(root, [p for r, p in files if r == root]))
        for root, path in files:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            rel = comp.rel(root, path)
            doc_id = f"{comp.name}_{re.sub(r'[^a-z0-9_]', '_', rel.lower())}"
            secs = sections_of(text, rst=path.suffix.lower() == ".rst", adoc=path.suffix.lower() in (".asc", ".adoc"))
            title = next((p[0] for _, p, *_ in secs if p), path.stem)
            db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)", (doc_id, comp.name, rel, str(path),
                       hashlib.sha256(raw).hexdigest(), times.get(path, 0), len(text.split()), title))
            s = stats[comp.name]
            s["docs"] += 1
            for k, (level, hpath, a, b, body) in enumerate(secs):
                sid = f"{doc_id}_s{k}"
                mentions = set(BACKTICK.findall(body)) | set(PATHLIKE.findall(body)) | set(CODEWORD.findall(body))
                resolved, unresolved = 0, 0
                for mtn in mentions:
                    key, cands = resolve(mtn, idx, comp_of, comp.name)
                    if cands and len(cands) <= 3:
                        conf, score = ("EXTRACTED", 1.0) if len(cands) == 1 else ("AMBIGUOUS", 0.3)
                        for c in cands:
                            db.execute("INSERT INTO links VALUES (?,?,?,?,?)", (sid, c, key, conf, score))
                        resolved += 1
                    elif not cands and mtn in BACKTICK.findall(body) and CODE_SHAPED.search(mtn) and " " not in mtn:
                        db.execute("INSERT INTO unresolved VALUES (?,?)", (sid, mtn))
                        unresolved += 1
                db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (sid, doc_id, comp.name, rel, " > ".join([path.name] + hpath), level, k, a, b, body,
                            len(body.split()), len(mentions), resolved, unresolved))
                s["sections"] += 1; s["resolved"] += resolved; s["unresolved"] += unresolved; s["words"] += len(body.split())
    for comp in ws["components"]:
        for p in comp.skipped:
            print(f"skipped (iCloud, not downloaded): {p}  -> run `brctl download '{p}'` to include it")
    db.execute("INSERT INTO sections_fts(sections_fts) VALUES ('rebuild')")
    db.commit()
    n_links = db.execute("SELECT COUNT(*), COUNT(DISTINCT node_id) FROM links WHERE confidence='EXTRACTED'").fetchone()
    for c, s in stats.items():
        print(f"{c:8s} docs {s['docs']:4d} sections {s['sections']:5d} words {s['words']:7d} | mentions resolved "
              f"{s['resolved']:5d} unresolved {s['unresolved']:4d}")
    print(f"text KB -> {db_path}: {n_links[0]} exact doc->code links to {n_links[1]} distinct code nodes")
    (out / "text_kb_stats.json").write_text(json.dumps({c: dict(s) for c, s in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
