"""Query one built memory.

    from kev_memory import Memory, Models
    models = Models()                                   # loads Kev-Ret-B (+ Kev-Rerank lazily); share across memories
    mem = Memory("~/.kev-memory/acme", models)
    result = mem.search("how does the feed reach mobile", mode="fast", k=5)
    print(mem.context(result))                          # cited context pack for an LLM

Every hit is a section of a document, quoted verbatim with path and line range, plus the code nodes it names
(doc -> code links from the AST graph), so one query returns the text and the code it belongs to."""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev_memory import store
from kev_memory.retrieval import keyword, policy

PATH_OR_CALL = re.compile(r"(?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|jsx|mjs|go|rs|rb|java|kt|swift|sh|ya?ml|toml|sql|tf)"
                          r"|[A-Za-z_][\w.]*\(\)")


class Models:
    """Loads the models once; the reranker only when a query first needs it (it is the slow part)."""

    def __init__(self, weights: Path | None = None, device: str | None = None):
        self.weights, self.device = weights, device
        self._retriever = self._reranker = None

    @property
    def retriever(self):
        if self._retriever is None:
            from kev_memory.models.encoder import KevRetriever
            self._retriever = KevRetriever(self.weights, self.device)
        return self._retriever

    @property
    def reranker(self):
        if self._reranker is None:
            from kev_memory.models.reranker import KevReranker
            self._reranker = KevReranker(self.weights, self.device)
        return self._reranker


class Memory:
    def __init__(self, out_dir: str | Path, models: Models | None = None, name: str | None = None):
        self.out = Path(out_dir).expanduser()
        self.models = models or Models()
        self.manifest = json.loads((self.out / store.MANIFEST).read_text())
        self.name = name or self.manifest.get("workspace", self.out.name)
        self.db = store.open_readonly(self.out / store.KNOWLEDGE_DB)
        v = np.load(self.out / store.SECTION_VECTORS)
        self.section_ids = list(v["ids"])
        self.section_vecs = v["vecs"].astype(np.float32)
        self.section_index = {s: i for i, s in enumerate(self.section_ids)}
        code = json.loads((self.out / store.CODE_GRAPH).read_text())
        self.code_nodes = {n["id"]: n for n in code["nodes"]}
        self.file_times = code.get("file_times", {})
        self.code_links = defaultdict(list)
        for sid, nid in self.db.execute("SELECT section_id, node_id FROM links WHERE confidence='EXTRACTED'"):
            if nid in self.code_nodes:
                self.code_links[sid].append(nid)

    def section(self, sid: str) -> dict:
        row = self.db.execute(f"SELECT {', '.join(store.SECTION_FIELDS)} FROM sections WHERE id=?", (sid,)).fetchone()
        return dict(zip(store.SECTION_FIELDS, row))

    def candidates(self, query: str, qvec: np.ndarray | None = None):
        """-> (bm25 ids, dense ids, cosine of every candidate): the two first-stage lists, top 50 each."""
        bm25 = [s for s, _ in keyword.bm25_sections(self.db, query, policy.DEPTH)]
        qvec = self.models.retriever.embed_query(query) if qvec is None else qvec
        cos = self.section_vecs @ qvec
        top = np.argsort(-cos)[:policy.DEPTH]
        dense = [self.section_ids[i] for i in top]
        ix = self.section_index
        return bm25, dense, {s: float(cos[ix[s]]) for s in dict.fromkeys(bm25 + dense)}

    def rerank_scores(self, query: str, bm25: list[str], dense: list[str]) -> dict[str, float]:
        pool = policy.rerank_pool(bm25, dense)
        secs = [self.section(s) for s in pool]
        scores = self.models.reranker.score(query, [f"{s['heading_path']}\n{s['text']}" for s in secs])
        return {s: float(v) for s, v in zip(pool, scores)}

    def search(self, query: str, mode: str = "fast", k: int | str = 5, qvec: np.ndarray | None = None) -> dict:
        """-> {"query", "mode", "mode_used", "k", "hits": [...], "timing_ms"}. k="auto": adaptive k (1-5) from the
        calibrated dense confidence (policy.auto_k)."""
        t0 = time.time()
        bm25, dense, cos = self.candidates(query, qvec)
        t1 = time.time()
        scores = self.rerank_scores(query, bm25, dense) if policy.needs_reranker(mode, bm25, dense) else None
        t2 = time.time()
        ranked, used = policy.rank(mode, bm25, dense, scores)
        k = policy.auto_k(ranked, cos) if k == "auto" else int(k)
        hits = []
        for i, sid in enumerate(ranked[:k]):
            s = self.section(sid)
            code = [{"label": self.code_nodes[n].get("label"), "file": self.code_nodes[n].get("source_file"),
                     "location": self.code_nodes[n].get("source_location"), "kind": self.code_nodes[n].get("kind")}
                    for n in self.code_links.get(sid, [])[:6]]
            verify, stale = self.grounding(sid, s["path"])
            missing = self.missing_names(sid, s["path"])
            hits.append({"rank": i + 1, "domain": self.name, "section_id": sid, "path": s["path"],
                         "lines": [s["line_start"], s["line_end"]], "heading": s["heading_path"], "text": s["text"],
                         "rerank_score": None if scores is None else scores.get(sid), "cosine": cos.get(sid),
                         "code": code, "verify": verify, "stale": stale, "missing": missing})
        return {"query": query, "mode": mode, "mode_used": used, "k": k, "hits": hits,
                "timing_ms": {"first_stage": round(1000 * (t1 - t0)), "rerank": round(1000 * (t2 - t1))}}

    def grounding(self, sid: str, doc_path: str, n: int = 4) -> tuple[list[str], list[dict]]:
        """Files to check an answer against: the config and code files this section links to (config first, at most
        `n`), and the ones whose last commit is newer than the doc's, i.e. where the doc may be out of date."""
        def files_of(sids):
            nodes = [c for x in sids for c in self.code_links.get(x, [])]
            return list(dict.fromkeys(self.code_nodes[c]["source_file"] for c in sorted(
                nodes, key=lambda c: self.code_nodes[c].get("kind") != "config_file")
                if self.code_nodes[c].get("source_file")))[:n]
        files = files_of([sid])
        if not files:                               # section names no file: fall back to what its page links to
            files = files_of([x for (x,) in self.db.execute("SELECT id FROM sections WHERE path=? ORDER BY ordinal",
                                                             (doc_path,))])
        row = self.db.execute("SELECT time FROM docs WHERE path=?", (doc_path,)).fetchone()
        doc_t = row[0] if row and row[0] else None
        stale = [{"file": f, "changed": time.strftime("%Y-%m-%d", time.localtime(self.file_times[f])),
                  "doc": time.strftime("%Y-%m-%d", time.localtime(doc_t))}
                 for f in files if doc_t and self.file_times.get(f, 0) > doc_t]
        return files, stale

    def missing_names(self, sid: str, doc_path: str) -> list[str]:
        """Names the section uses that do not exist in the code: a call `name()`, or a file path whose directory
        exists in the repo (at its root or next to the doc) while the file does not. Strong signs that the doc
        describes code that was renamed or removed. Paths into other repos, env vars, URLs and prose words are left
        out: they are not expected to be in this code graph."""
        row = self.db.execute("SELECT abspath FROM docs WHERE path=?", (doc_path,)).fetchone()
        bases = [Path(r) for r in self.manifest.get("sources", {})] + ([Path(row[0]).parent] if row else [])
        out = []
        for (m,) in self.db.execute("SELECT mention FROM unresolved WHERE section_id=?", (sid,)):
            m = m.strip().strip("`")
            if not PATH_OR_CALL.fullmatch(m) or m.startswith(("http", "/")):
                continue
            if m.endswith("()") or any((b / m).parent.is_dir() and not (b / m).exists() for b in bases):
                out.append(m)
        calls = [m for m in out if m.endswith("()")]
        if calls:                                   # the AST misses some definitions (arrow consts, dynamic exports):
            found = self._in_code([c[:-2].rsplit(".", 1)[-1] for c in calls])     # a textual hit in code clears it
            out = [m for m in out if not (m.endswith("()") and m[:-2].rsplit(".", 1)[-1] in found)]
        return out[:5]

    def _in_code(self, names: list[str]) -> set[str]:
        """Which of `names` occur as whole words in non-doc files of the indexed repos (one `git grep` per repo)."""
        found = set()
        for root in self.manifest.get("sources", {}):
            args = ["git", "-C", root, "grep", "-h", "-o", "-w", "-I", "-F"]
            for n in names:
                args += ["-e", n]
            args += ["--", ".", ":!*.md", ":!*.mdx", ":!*.rst"]
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=5)
                found |= set(r.stdout.split())
            except (OSError, subprocess.SubprocessError):
                found |= set(names)                 # cannot check: do not claim a conflict
        return found

    def docs_for(self, path: str) -> list[dict]:
        """Doc sections that link to a code or config file (any node in it): the docs to review after changing it."""
        path = path.strip().removeprefix("./")
        nodes = [n for n, v in self.code_nodes.items()
                 if (v.get("source_file") or "").endswith("/" + path) or v.get("source_file") == path]
        if not nodes:
            return []
        q = f"SELECT DISTINCT l.section_id FROM links l WHERE l.node_id IN ({','.join('?' * len(nodes))})"
        out = []
        for (sid,) in self.db.execute(q, nodes):
            s = self.section(sid)
            out.append({"domain": self.name, "path": s["path"], "lines": [s["line_start"], s["line_end"]],
                        "heading": s["heading_path"]})
        return sorted(out, key=lambda h: (h["path"], h["lines"][0]))

    @staticmethod
    def context(result: dict, budget_tokens: int = 2000, words_per_hit: int = 300) -> str:
        """Cited context pack: one block per hit ('[n] path:Lx-y  heading' + text + linked code), ~4 chars/token."""
        parts, used = [], 0
        for h in result["hits"]:
            body = " ".join(h["text"].split()[:words_per_hit])
            code = "".join(f"\n  -> code {c['label']} ({c['file']}{':' + c['location'] if c['location'] else ''})"
                           for c in h["code"])
            code += "".join(f"\n  ! doc may be stale: {x['file']} changed {x['changed']}, after this doc ({x['doc']})"
                            for x in h.get("stale", []))
            if h.get("missing"):
                code += "\n  ! doc/code conflict: names not found in the code: " + ", ".join(f"`{m}`" for m in h["missing"])
            if h.get("verify"):
                code += "\n  verify in: " + ", ".join(h["verify"])
            block = f"[{h['rank']}] {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}\n{body}{code}\n"
            if used + len(block) // 4 > budget_tokens:
                break
            parts.append(block)
            used += len(block) // 4
        return "\n".join(parts)

    def status(self) -> dict:
        return {"name": self.name, "out": str(self.out), **{k: self.manifest.get(k) for k in
                ("built_at", "encoder", "counts", "sources", "changed")}}
