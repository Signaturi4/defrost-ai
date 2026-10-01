"""Doc -> code links, "exact symbol match first": a section links to a code node when one of its mentions names
exactly one node (preferring the document's own component). Mentions are backticked spans, file paths and
code-shaped words (camelCase, snake_case, dotted Terraform refs). 2-3 candidates are kept as AMBIGUOUS; more is too
generic to use. Backticked code-shaped mentions that resolve to nothing are recorded as unresolved (stale or external)."""
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


def mentions(body: str) -> tuple[set[str], set[str]]:
    """-> (all mentions, backticked mentions)"""
    ticks = set(BACKTICK.findall(body))
    return ticks | set(PATHLIKE.findall(body)) | set(CODEWORD.findall(body)), ticks


def symbol_index(nodes: list[dict]) -> dict[str, set[str]]:
    """name -> node ids. Keys: label without call parens, file name/path tail for file nodes, dotted Terraform key."""
    idx = defaultdict(set)
    for n in nodes:
        lab = (n.get("label") or "").strip()
        if not lab or len(lab) > 80 or n.get("kind") == "rationale" or n.get("file_type") == "rationale":
            continue
        key = (lab[:-2] if lab.endswith("()") else lab).lstrip(".")
        if len(key) >= 3:
            idx[key].add(n["id"])
        src = n.get("source_file") or ""
        if src and lab == Path(src).name:                     # file node: also index its path tail
            parts = src.split("/")[1:]                        # drop the component prefix
            for k in range(1, min(4, len(parts)) + 1):
                idx["/".join(parts[-k:])].add(n["id"])
    return idx


def resolve(mention: str, idx, component_of: dict[str, str], component: str):
    """-> (normalised mention, candidate node ids)"""
    m = mention.strip().strip("`'\"").rstrip(".,:;")
    m = m[:-2] if m.endswith("()") else m
    cands = idx.get(m) or idx.get(m.split("(")[0]) or idx.get(m.rsplit(".", 1)[-1] if "." in m and "/" not in m
                                                              and not m.startswith(("var.", "module.", "aws_")) else "")
    if not cands:
        return None, []
    own = [c for c in cands if component_of[c] == component]
    return m, sorted(own or cands)


def link_section(body: str, idx, component_of, component):
    """-> (links [(node_id, mention, confidence, score)], unresolved [mention], n_mentions, n_resolved)"""
    found, ticks = mentions(body)
    links, unresolved, n_resolved = [], [], 0
    for mtn in found:
        key, cands = resolve(mtn, idx, component_of, component)
        if cands and len(cands) <= 3:
            conf, score = ("EXTRACTED", 1.0) if len(cands) == 1 else ("AMBIGUOUS", 0.3)
            links += [(c, key, conf, score) for c in cands]
            n_resolved += 1
        elif not cands and mtn in ticks and CODE_SHAPED.search(mtn) and " " not in mtn:
            unresolved.append(mtn)
    return links, unresolved, len(found), n_resolved
