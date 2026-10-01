"""Doc -> code links, "exact symbol match first": a section links to a code node when one of its mentions names
exactly one node (preferring the document's own component). Mentions are backticked spans, file paths and
code-shaped words (camelCase, snake_case, dotted Terraform refs). 2-3 candidates are kept as AMBIGUOUS; more is too
generic to use. Backticked code-shaped mentions that resolve to nothing are recorded as unresolved (stale or external).

Precision rules (measured on 100 hand-labelled links; see scripts/link_audit.py):
- Link targets are top-level functions/classes, methods and files. Test/spec/example files and nested functions are
  only reachable through an explicit file path; nodes without a source file (builtins) never.
- Methods are indexed as `Class.method`; the bare method name only answers code-shaped mentions (`grantedScopes`).
- A one-word mention is EXTRACTED only when its target is top-level and outside tests; a lowercase word
  (`pending`, `worker`, `error`) also needs the target in the project core and its file name in the same section.
  Otherwise it is AMBIGUOUS (stored, not shown): such words are usually status values, roles, columns or packages.
- A dotted name may fall back to its last part (`fields.Tuple` -> `Tuple`) only when an earlier part matches the
  target's class, file or folder.
- Core: with `core` paths (workspace `core: [...]`, or detected: server-side folders such as backend/, api/, db/,
  pipeline/, else src/), an ambiguous mention with exactly one core candidate links to it, and links are shown core
  first, then by specificity (path > qualified name > code-shaped > bare word)."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

BACKTICK = re.compile(r"`([^`\n]{2,120})`")
PATHLIKE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*(?:[\w.-]+\.(?:py|ts|tsx|js|jsx|mjs|tf|swift|kt|go|ya?ml|json|sh|sql|toml"
                      r"|ini|cfg|conf|Dockerfile|dockerfile)|(?:[\w.-]+/)*(?:Dockerfile|Makefile|Procfile|crontab|Caddyfile)))\b")
CODEWORD = re.compile(r"\b([a-z]+[A-Z][A-Za-z0-9]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+|[a-z][a-z0-9]*_[a-z0-9_]{2,}"
                      r"|(?:var|module|data|aws_[a-z0-9_]+)\.[A-Za-z0-9_.-]+)\b")
CODE_SHAPED = re.compile(r"[()_./]|[a-z][A-Z]|^[A-Z][a-z]+[A-Z]")
BARE_WORD = re.compile(r"[A-Za-z][a-z0-9]*")                 # one plain word: `pending`, `Session`, `worker`
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec|specs|e2e|fixtures?|testing)(/|$)|[._-](test|spec)\.[a-z]+$"
                       r"|(^|/)(test_[^/]+|conftest\.py)$")
EXAMPLE_PATH = re.compile(r"(^|/)(examples?|samples?)(/|$)")
NONCORE_DIRS = {"test", "tests", "__tests__", "spec", "e2e", "fixtures", "examples", "example", "docs", "doc", "scripts",
                "tools", "benchmarks", "migrations", "stories", "storybook"}
CORE_STRONG = {"backend", "server", "api", "services", "service", "worker", "workers", "jobs", "pipeline", "db",
               "database", "core", "internal", "cmd", "scraper"}
CORE_WEAK = {"src", "lib", "pkg"}
SPECIFICITY = ("path", "qualified", "code", "word")


def specificity(mention: str) -> int:
    m = mention.strip("`")
    if "/" in m or PATHLIKE.fullmatch(m):
        return 0
    if "." in m:
        return 1
    return 3 if BARE_WORD.fullmatch(m) else 2


def core_paths(spec_or_ws, nodes: list[dict]) -> list[str] | None:
    """Path prefixes ('<component>/<dir>/') of the project core. Explicit `core: [...]` on a workspace component wins;
    otherwise the shallowest server-side folders (CORE_STRONG names) that hold code, else src/lib/pkg folders.
    Folders under tests, examples, docs, scripts or tools never count. None = no rule (everything is core)."""
    comps = spec_or_ws.get("components", []) if isinstance(spec_or_ws, dict) else \
        [{"name": c.name, "core": getattr(c, "core", None)} for c in getattr(spec_or_ws, "components", [])]
    explicit = [f"{c['name']}/{p.strip('/')}/" for c in comps for p in (c.get("core") or [])]
    if explicit:
        return explicit
    dirs = set()
    for n in nodes:
        f = n.get("source_file") or ""
        if not f or TEST_PATH.search(f):
            continue
        parts = f.split("/")[:-1]
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i + 1]))
    def pick(names):
        hits = sorted((d for d in dirs if d.rsplit("/", 1)[-1] in names
                       and not any(x in NONCORE_DIRS or x.startswith(".") for x in d.split("/")[1:])), key=len)
        out = []
        for d in hits:
            if not any(d.startswith(o) for o in out):
                out.append(d + "/")
        return out
    return pick(CORE_STRONG) or pick(CORE_WEAK) or None


def in_core(source_file: str, core) -> bool:
    return core is None or any((source_file or "").startswith(c) for c in core)


class SymbolIndex(defaultdict):
    """name -> node ids, plus per-node facts: role ('file' | 'top' | 'method' | 'nested'), class (for methods),
    test (in a test/spec file), and bare method names (`methods`) that only code-shaped mentions may use."""
    def __init__(self):
        super().__init__(set)
        self.info: dict[str, dict] = {}
        self.methods: dict[str, set[str]] = defaultdict(set)


def mentions(body: str) -> tuple[set[str], set[str]]:
    """-> (all mentions, backticked mentions)"""
    ticks = set(BACKTICK.findall(body))
    return ticks | set(PATHLIKE.findall(body)) | set(CODEWORD.findall(body)), ticks


def symbol_index(nodes: list[dict], edges: list[dict] | None = None) -> SymbolIndex:
    """name -> node ids. Keys: label without call parens (methods as Class.method), file name/path tail for file
    nodes, dotted Terraform key. Without `edges` every symbol counts as top-level (no nesting information)."""
    idx = SymbolIndex()
    by_id = {n["id"]: n for n in nodes}
    parent = {}
    for e in edges or []:
        if e.get("relation") in ("contains", "method") and e.get("target") in by_id:
            parent[e["target"]] = (e["relation"], by_id.get(e["source"], {}).get("label", ""))
    for n in nodes:
        lab = (n.get("label") or "").strip()
        if not lab or len(lab) > 80 or n.get("kind") == "rationale" or n.get("file_type") == "rationale":
            continue
        src = n.get("source_file") or ""
        is_file = bool(src) and lab == Path(src).name
        rel, owner = parent.get(n["id"], (None, ""))
        role = "file" if is_file else "method" if rel == "method" else "top" if (rel or edges is None) else "nested"
        test = bool(TEST_PATH.search(src) or EXAMPLE_PATH.search(src))   # tests and examples: by path only
        idx.info[n["id"]] = {"role": role, "class": owner if rel == "method" else "", "test": test, "file": src}
        if is_file:                                           # file node: its path tails, tests included
            parts = src.split("/")[1:]                        # drop the component prefix
            for k in range(1, min(4, len(parts)) + 1):
                idx["/".join(parts[-k:])].add(n["id"])
            continue
        if not src or test or role == "nested":               # builtins, tests, nested: reachable by path only
            continue
        key = (lab[:-2] if lab.endswith("()") else lab).lstrip(".")
        if len(key) < 3:
            continue
        if role == "method" and owner:
            idx[f"{owner}.{key}"].add(n["id"])
            idx.methods[key].add(n["id"])
        else:
            idx[key].add(n["id"])
    return idx


def _owner_tokens(info: dict) -> set[str]:
    toks = {t for part in info.get("file", "").split("/") for t in re.split(r"[._-]", part.lower()) if len(t) >= 3}
    return toks | ({info["class"].lower()} if info.get("class") else set())


def _prefix_matches(prefix: list[str], info: dict) -> bool:
    toks = _owner_tokens(info)
    cls = (info.get("class") or "").lower()
    for p in prefix:
        p = p.lower().lstrip("~")
        if len(p) >= 3 and (p in toks or (cls and p in cls)):
            return True
    return False


def resolve(mention: str, idx, component_of: dict[str, str], component: str):
    """-> (normalised mention, candidate node ids, how) with how in exact | method | dotted"""
    m = mention.strip().strip("`'\"").rstrip(".,:;")
    m = m[:-2] if m.endswith("()") else m
    m = m.split("(")[0] if "(" in m else m
    how, cands = "exact", set(idx.get(m, ()))
    if not cands and CODE_SHAPED.search(m) and not BARE_WORD.fullmatch(m) and "." not in m:
        how, cands = "method", set(getattr(idx, "methods", {}).get(m, ()))
    if not cands and "." in m and "/" not in m and not m.startswith(("var.", "module.", "aws_")):
        parts = m.split(".")
        last = parts[-1]
        pool = set(idx.get(last, ())) | set(getattr(idx, "methods", {}).get(last, ()))
        info = getattr(idx, "info", None)
        if info is not None:                                   # the earlier parts must name the class / file / folder
            pool = {c for c in pool if _prefix_matches(parts[:-1], info.get(c, {}))}
        how, cands = "dotted", pool
    if not cands:
        return None, [], how
    own = [c for c in cands if component_of[c] == component]
    return m, sorted(own or cands), how


def _bare_ok(nid: str, mention: str, body: str, idx, core) -> bool:
    """A one-word mention links only to a top-level symbol outside tests. A lowercase word (`pending`, `worker`,
    `error`) also needs the target in the project core and its file name (with extension) in the same section:
    lowercase words in docs are mostly status values, roles, columns and package names. Capitalised words are
    class / component names and need no more."""
    info = getattr(idx, "info", {}).get(nid)
    if info is None:
        return True                                            # no facts (plain dict index): old behaviour
    if info["role"] != "top" or info["test"]:
        return False
    if mention[:1].isupper():
        return True
    name = info["file"].rsplit("/", 1)[-1]
    return in_core(info["file"], core) and name in body


def link_section(body: str, idx, component_of, component, core=None):
    """-> (links [(node_id, mention, confidence, score)], unresolved [mention], n_mentions, n_resolved).
    Links come back ordered for display: core first, then by specificity (path > qualified > code-shaped > word)."""
    found, ticks = mentions(body)
    links, unresolved, n_resolved = [], [], 0
    info = getattr(idx, "info", {})
    for mtn in sorted(found):
        key, cands, how = resolve(mtn, idx, component_of, component)
        if cands and len(cands) > 1 and core is not None and specificity(key) > 0:   # not for paths: one in core wins
            in_c = [c for c in cands if in_core(info.get(c, {}).get("file", ""), core)]
            if len(in_c) == 1:
                cands = in_c
        if cands and len(cands) <= 3:
            conf, score = ("EXTRACTED", 1.0) if len(cands) == 1 else ("AMBIGUOUS", 0.3)
            if conf == "EXTRACTED" and BARE_WORD.fullmatch(key) and not _bare_ok(cands[0], key, body, idx, core):
                conf, score = "AMBIGUOUS", 0.3
            links += [(c, key, conf, score) for c in cands]
            n_resolved += 1
        elif not cands and mtn in ticks and CODE_SHAPED.search(mtn) and " " not in mtn:
            unresolved.append(mtn)
    links.sort(key=lambda l: (l[2] != "EXTRACTED", not in_core(info.get(l[0], {}).get("file", ""), core),
                              specificity(l[1])))
    return links, unresolved, len(found), n_resolved
