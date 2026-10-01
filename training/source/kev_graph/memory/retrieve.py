"""Combined retrieval over the two KBs, weighted by doc trust, expanded through the graph.

    uv run python -m kev_graph.memory.retrieve --workspace kev_graph/memory/workspaces/client.json "how does the feed reach mobile"

Four ranked lists, fused by reciprocal rank (RRF, k=60):
  text/bm25   FTS5 BM25 over section heading paths + text
  text/dense  cosine between the query and section vectors (CGSA encoder, embed.py)
  code/bm25   FTS5 over code node labels and paths, camelCase/snake_case split into words
  code/dense  cosine against code node vectors
Then weights (trust.py): a text hit scores x 2*weight(section), a code hit x 1.0. Then graph expansion: the top text
hits pull in the code nodes they link to (exact doc->code links), the top code hits pull in the sections that link to
them and their 1-hop code neighbours. Every result carries a citation (path + line) for grounding.

`mode` switches parts off for ablations: bm25 | dense | text | code | hybrid | graph (= hybrid + expansion, default)."""
import argparse
import json
import re
import sqlite3
from collections import defaultdict

import numpy as np

from kev_graph.memory.workspace import load

STOP = set("a an and are as at be by can do does for from has have how i in is it its of on or the this to was what "
           "when where which who why will with there their they them into via any all our we you your".split())
MODES = ("bm25", "dense", "text", "code", "hybrid", "graph")


def words(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if len(w) > 1]


def fts_query(q):
    ws = [w for w in dict.fromkeys(words(q)) if w not in STOP]
    return " OR ".join(f'"{w}"' for w in ws)


class Memory:
    def __init__(self, workspace, encoder=None, encoder_name=None):
        from kev_graph.memory.embed import vectors_file
        ws = load(workspace)
        self.out = ws["out"]
        self.encoder_name = encoder_name or ws.get("encoder", "cgsa")
        self.db = sqlite3.connect(self.out / "text_kb.sqlite")
        code = json.loads((self.out / "code_kb.json").read_text())
        self.nodes = {n["id"]: n for n in code["nodes"]}
        self.adj = defaultdict(list)
        for e in code["edges"]:
            if e["relation"] in ("calls", "contains", "references", "uses", "imports_from", "inherits", "method"):
                self.adj[e["source"]].append(e["target"])
                self.adj[e["target"]].append(e["source"])
        self.links = defaultdict(list)          # section -> [node]
        self.backlinks = defaultdict(list)      # node -> [section]
        for sid, nid, conf in self.db.execute("SELECT section_id, node_id, confidence FROM links"):
            if conf == "EXTRACTED":
                self.links[sid].append(nid)
                self.backlinks[nid].append(sid)
        self.trust = json.loads((self.out / "trust.json").read_text())
        # code FTS, in memory: labels and paths split into words so "recsys service" finds recsys.service.ts
        self.cdb = sqlite3.connect(":memory:")
        self.cdb.execute("CREATE VIRTUAL TABLE c USING fts5(id UNINDEXED, words, tokenize='porter unicode61')")
        self.cdb.executemany("INSERT INTO c VALUES (?, ?)",
                             [(i, " ".join(words(n.get("label", "")) + words(n.get("source_file", ""))))
                              for i, n in self.nodes.items() if (n.get("kind") or n.get("file_type")) != "rationale"])
        v = np.load(self.out / vectors_file(self.encoder_name))
        self.sec_ids, self.sec_vecs = list(v["section_ids"]), v["section_vecs"].astype(np.float32)
        self.node_ids, self.node_vecs = list(v["node_ids"]), v["node_vecs"].astype(np.float32)
        self._encoder = encoder

    def encoder(self):
        if self._encoder is None:
            from kev_graph.memory.embed import get_encoder
            self._encoder = get_encoder(self.encoder_name)
        return self._encoder

    def section(self, sid):
        r = self.db.execute("SELECT path, heading_path, line_start, line_end, text, weight, component FROM sections "
                            "WHERE id=?", (sid,)).fetchone()
        return dict(zip(["path", "heading_path", "line_start", "line_end", "text", "weight", "component"], r))

    def _lists(self, q, mode, n):
        lists = {}
        fq = fts_query(q)
        if mode in ("bm25", "text", "hybrid", "graph") and fq:
            lists["text/bm25"] = [("text", r[0]) for r in self.db.execute(
                "SELECT s.id FROM sections_fts f JOIN sections s ON s.rowid=f.rowid WHERE sections_fts MATCH ? "
                "ORDER BY bm25(sections_fts, 2.0, 1.0) LIMIT ?", (fq, n))]
        if mode in ("bm25", "code", "hybrid", "graph") and fq:
            lists["code/bm25"] = [("code", r[0]) for r in self.cdb.execute(
                "SELECT id FROM c WHERE c MATCH ? ORDER BY bm25(c) LIMIT ?", (fq, n))]
        if mode in ("dense", "text", "code", "hybrid", "graph"):
            qv = self.encoder().query([q])[0]
            if mode != "code":
                top = np.argsort(-(self.sec_vecs @ qv))[:n]
                lists["text/dense"] = [("text", self.sec_ids[i]) for i in top]
            if mode not in ("text",):
                top = np.argsort(-(self.node_vecs @ qv))[:n]
                lists["code/dense"] = [("code", self.node_ids[i]) for i in top]
        return lists

    def search(self, q, k=10, mode="graph", lam=None, n=50):
        """-> ranked [{kind, id, path, loc, title, score, via}]. lam overrides every component's lambda."""
        assert mode in MODES
        lists = self._lists(q, mode, n)
        score, via = defaultdict(float), defaultdict(set)
        for name, items in lists.items():
            for rank, key in enumerate(items):
                score[key] += 1.0 / (60 + rank + 1)
                via[key].add(name)
        for key in list(score):
            if key[0] == "text":
                s = self.section(key[1])
                w = s["weight"] if lam is None else lam * s["weight"] / self.trust[s["component"]]["lambda"]
                score[key] *= 2 * w
        if mode == "graph":
            ranked = sorted(score, key=score.get, reverse=True)
            for key in ranked[:8]:
                base = score[key]
                if key[0] == "text":
                    for nid in self.links.get(key[1], [])[:6]:
                        k2 = ("code", nid)
                        score[k2] = max(score[k2], 0.5 * base); via[k2].add(f"link<-{key[1]}")
                else:
                    for sid in self.backlinks.get(key[1], [])[:4]:
                        k2 = ("text", sid)
                        score[k2] = max(score[k2], 0.5 * base); via[k2].add(f"backlink<-{key[1]}")
                    for nid in self.adj.get(key[1], [])[:6]:
                        k2 = ("code", nid)
                        score[k2] = max(score[k2], 0.3 * base); via[k2].add(f"edge<-{key[1]}")
        out = []
        for key in sorted(score, key=score.get, reverse=True)[:k]:
            if key[0] == "text":
                s = self.section(key[1])
                out.append({"kind": "text", "id": key[1], "path": s["path"], "loc": f"L{s['line_start']}-{s['line_end']}",
                            "title": s["heading_path"], "text": s["text"], "score": score[key], "via": sorted(via[key])})
            else:
                nd = self.nodes.get(key[1], {})
                out.append({"kind": "code", "id": key[1], "path": nd.get("source_file", ""), "loc": nd.get("source_location"),
                            "title": nd.get("label", ""), "text": f"{nd.get('label', '')} [{nd.get('kind') or nd.get('file_type')}]",
                            "score": score[key], "via": sorted(via[key])})
        return out

    @staticmethod
    def context(results, budget_tokens=2000):
        """Grounded context pack: each item cited by path:line, cut to ~budget tokens (4 chars/token)."""
        parts, used = [], 0
        for r in results:
            body = r["text"] if r["kind"] == "text" else r["title"]
            block = f"[{r['kind']}] {r['path']}:{r['loc']}  {r['title']}\n{body}\n"
            cost = len(block) // 4
            if used + cost > budget_tokens:
                block = block[:max(0, (budget_tokens - used) * 4)]
                cost = len(block) // 4
            if cost <= 0:
                break
            parts.append(block); used += cost
        return "\n".join(parts), used


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--mode", default="graph", choices=MODES)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("query")
    args = ap.parse_args(argv)
    mem = Memory(args.workspace)
    res = mem.search(args.query, k=args.k, mode=args.mode)
    for r in res:
        print(f"{r['score']:.4f} [{r['kind']}] {r['path']}:{r['loc']}  {r['title'][:90]}  via {','.join(r['via'])[:60]}")
    print("\n--- context pack ---\n" + mem.context(res, args.budget)[0])


if __name__ == "__main__":
    main()
