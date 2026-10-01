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
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev_memory import store
from kev_memory.retrieval import keyword, policy


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
                     "location": self.code_nodes[n].get("source_location")} for n in self.code_links.get(sid, [])[:6]]
            hits.append({"rank": i + 1, "domain": self.name, "section_id": sid, "path": s["path"],
                         "lines": [s["line_start"], s["line_end"]], "heading": s["heading_path"], "text": s["text"],
                         "rerank_score": None if scores is None else scores.get(sid), "cosine": cos.get(sid),
                         "code": code})
        return {"query": query, "mode": mode, "mode_used": used, "k": k, "hits": hits,
                "timing_ms": {"first_stage": round(1000 * (t1 - t0)), "rerank": round(1000 * (t2 - t1))}}

    @staticmethod
    def context(result: dict, budget_tokens: int = 2000, words_per_hit: int = 300) -> str:
        """Cited context pack: one block per hit ('[n] path:Lx-y  heading' + text + linked code), ~4 chars/token."""
        parts, used = [], 0
        for h in result["hits"]:
            body = " ".join(h["text"].split()[:words_per_hit])
            code = "".join(f"\n  -> code {c['label']} ({c['file']}{':' + c['location'] if c['location'] else ''})"
                           for c in h["code"])
            block = f"[{h['rank']}] {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}\n{body}{code}\n"
            if used + len(block) // 4 > budget_tokens:
                break
            parts.append(block)
            used += len(block) // 4
        return "\n".join(parts)

    def status(self) -> dict:
        return {"name": self.name, "out": str(self.out), **{k: self.manifest.get(k) for k in
                ("built_at", "encoder", "counts", "sources", "changed")}}
