"""P2b S3 on Kaggle: CodeRAG-Bench (ODEX, DS-1000 vs the 34k library-documentation pages) for Defrost-Ret arms.

Same protocol as defrost_graph/memory/public_ir.py: BM25 = SQLite FTS5 porter OR-query with bm25(2, 1) over (title, body);
dense = Defrost-Ret (MNTP [+CGSA] + supervised LoRA merged, bidirectional, mean pool, the tech-doc instruction on queries
and excluded from the pool), documents <= 256 tokens; hybrid = RRF(k=60). Metrics via ranx; paired bootstrap on
per-query nDCG@10 vs BM25. Nothing is tuned on these queries.

    python eval_public_sup.py --data DIR --arms A:final,B:final+cgsa,C:final --out DIR"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

INSTRUCTION = "Given a developer question about a software project, retrieve the documentation passage that answers it"
STOP = set("a an and are as at be by can do does for from has have how i in is it its of on or the this to was what "
           "when where which who why will with there their they them into via any all our we you your".split())


def words(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if len(w) > 1]


def fts_query(q):
    ws = [w for w in dict.fromkeys(words(q)) if w not in STOP]
    return " OR ".join(f'"{w}"' for w in ws)


def rrf(lists, k=60):
    s = {}
    for items in lists:
        for r, x in enumerate(items):
            s[x] = s.get(x, 0.0) + 1.0 / (k + r + 1)
    return s


def boot(a, b, n=5000, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


class DefrostRet:
    def __init__(self, mntp, cgsa, adapter, device, max_tokens=256):
        from transformers import AutoModel, AutoTokenizer
        from defrost_graph.bilm.bridge import _merge
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        m = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32, attn_implementation="sdpa")
        m = _merge(m, mntp)
        if cgsa:
            m = _merge(m, cgsa)
        self.model = _merge(m, adapter).to(device).half().eval()
        self.device, self.max_tokens = device, max_tokens

    @torch.no_grad()
    def encode(self, texts, prefix=None, bs=64):
        n_pre = len(self.tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for b in range(0, len(texts), bs):
            idx = order[b:b + bs]
            enc = self.tok([(prefix or "") + texts[i] for i in idx], padding=True, truncation=True,
                           max_length=self.max_tokens + n_pre, return_tensors="pt", padding_side="right").to(self.device)
            keep = enc["attention_mask"].bool()
            B, L = keep.shape
            mask = torch.zeros(B, 1, L, L, device=self.device, dtype=torch.float16)
            mask = mask.masked_fill(~keep[:, None, None, :], torch.finfo(torch.float16).min)
            h = self.model(input_ids=enc["input_ids"], attention_mask=mask).last_hidden_state.float()
            pool = keep.clone()
            pool[:, :n_pre] = False
            v = (h * pool[..., None]).sum(1) / pool.sum(1, keepdim=True).clamp(min=1)
            out[idx] = F.normalize(v, dim=-1).cpu().numpy()
        return out


def main(argv=None):
    from ranx import Qrels, Run, evaluate
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", default="A:final,B:final+cgsa,C:final")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    D = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(D / "public_ir/library-documentation.parquet")
    ids = list(corpus.doc_id)
    texts = [f"{t}\n{b}" for t, b in zip(corpus.doc_id, corpus.doc_content)]
    tasks = {}
    o = pd.read_parquet(D / "public_ir/odex.parquet")
    tasks["odex"] = [(f"odex-{i}", r.intent, {d["title"] for d in r.docs}) for i, r in enumerate(o.itertuples()) if len(r.docs)]
    s = pd.read_parquet(D / "public_ir/ds1000.parquet")
    tasks["ds1000"] = [(f"ds-{i}", r.prompt, {d["title"] for d in r.docs}) for i, r in enumerate(s.itertuples()) if len(r.docs)]
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE d USING fts5(title, body, tokenize='porter unicode61')")
    db.executemany("INSERT INTO d(rowid, title, body) VALUES (?,?,?)",
                   [(i, t.replace(".", " ").replace("_", " "), b) for i, (t, b) in enumerate(zip(corpus.doc_id, corpus.doc_content))])
    bm25 = lambda q: [ids[r[0]] for r in db.execute("SELECT rowid FROM d WHERE d MATCH ? ORDER BY bm25(d, 2.0, 1.0) "
                                                   "LIMIT 100", (fts_query(q),))] if fts_query(q) else []
    device = torch.device("cuda")
    report = {}
    runs = {t: {"bm25 (v1)": {qid: {x: 1 / (r + 1) for r, x in enumerate(bm25(q))} for qid, q, _ in items}}
            for t, items in tasks.items()}
    for spec in args.arms.split(","):
        arm, ck = spec.split(":")
        cg = ck.endswith("+cgsa")
        ck = ck.removesuffix("+cgsa")
        adapter = D / f"adapters/sup-{arm}/{ck}"
        t0 = time.time()
        enc = DefrostRet(D / "mntp-adapter", D / "cgsa-adapter" if cg else None, adapter, device)
        V = enc.encode(texts)
        print(f"arm {arm}: corpus encoded in {time.time() - t0:.0f}s", flush=True)
        for t, items in tasks.items():
            Q = enc.encode([q for _, q, _ in items], prefix=f"{INSTRUCTION}: ", bs=32)
            dn, hy = {}, {}
            for i, (qid, q, _) in enumerate(items):
                d = [ids[j] for j in np.argsort(-(V @ Q[i]))[:100]]
                dn[qid] = {x: 1 / (r + 1) for r, x in enumerate(d)}
                hy[qid] = rrf([list(runs[t]["bm25 (v1)"][qid]), d])
            runs[t][f"dense defrostret-{arm}"] = dn
            runs[t][f"hybrid bm25+defrostret-{arm}"] = hy
        del enc; torch.cuda.empty_cache()
    metrics = ["ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10", "map@10"]
    for t, items in tasks.items():
        qrels = Qrels({qid: {g: 1 for g in gold} for qid, _, gold in items})
        rs = {n: {qid: (r[qid] or {"__none__": 0.0}) for qid, _, _ in items} for n, r in runs[t].items()}
        per = {n: evaluate(qrels, Run(r, name=n), "ndcg@10", return_mean=False) for n, r in rs.items()}
        report[t] = {}
        print(f"\n{t}: {len(items)} queries")
        for n, r in rs.items():
            m = evaluate(qrels, Run(r, name=n), metrics)
            d = boot(per["bm25 (v1)"], per[n])
            report[t][n] = {**{k: round(float(v), 4) for k, v in m.items()}, "delta_ndcg10_vs_bm25": d}
            print(f"{n:28s} " + " ".join(f"{k}={m[k]:.3f}" for k in metrics) + f"  d={d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    (out / "public_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
