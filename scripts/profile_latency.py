"""Where does a search spend its time? Times every stage of one query separately (GPU synced), cold and warm.
Latency only: no quality metrics.

    KEV_MEMORY_MODELS=<weights> python scripts/profile_latency.py --memory ~/.kev-memory/parity-heldout \
        --questions benchmarks/heldout/questions.jsonl --n 20
-> prints a table; --out writes the raw timings as JSON."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

T0 = time.perf_counter()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402,F401
IMPORT_S = time.perf_counter() - T0

from kev_memory.memory import Memory, Models  # noqa: E402
from kev_memory.models.weights import RETRIEVAL_INSTRUCTION  # noqa: E402
from kev_memory.retrieval import keyword, policy  # noqa: E402


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


class Clock:
    def __init__(self, dev):
        self.dev, self.t = dev, {}

    def __call__(self, name, fn, *a, **kw):
        sync(self.dev)
        t = time.perf_counter()
        out = fn(*a, **kw)
        sync(self.dev)
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t) * 1000
        return out


def one_query(mem: Memory, q: str, mode: str, batch_size: int):
    ret, rr = mem.models.retriever, mem.models.reranker
    c = Clock(ret.device)
    bm25 = [s for s, _ in c("bm25_fts5", keyword.bm25_sections, mem.db, q, policy.DEPTH)]
    prefix = f"{RETRIEVAL_INSTRUCTION}: "
    enc = c("query_tokenize", ret.tok, [prefix + q], return_tensors="pt")
    qvec = c("query_embed_forward", ret.embed_query, q)              # includes its own tokenize (~0.3 ms)
    cos = c("dense_matmul", lambda: mem.section_vecs @ qvec)
    top = c("dense_topk", lambda: np.argsort(-cos)[:policy.DEPTH])
    dense = [mem.section_ids[i] for i in top]
    out = {"tokens_query": int(enc["input_ids"].shape[1])}
    if mode == "rerank" or policy.needs_reranker(mode, bm25, dense):
        pool = policy.rerank_pool(bm25, dense)
        secs = c("fetch_pool_sections", lambda: [mem.section(s) for s in pool])
        texts = [f"{s['heading_path']}\n{s['text']}" for s in secs]
        lens = []
        for b in range(0, len(texts), batch_size):
            ids, att = c("rerank_tokenize", rr._pairs, q, texts[b:b + batch_size])
            lens += att.sum(1).tolist()
            ids, att = c("rerank_to_device", lambda: (ids.to(rr.device), att.to(rr.device)))
            c("rerank_forward", lambda: rr.model(ids, att).float().cpu())
            out.setdefault("batch_shapes", []).append(list(ids.shape))
        out.update(pool=len(pool), pair_tokens_mean=float(np.mean(lens)), pair_tokens_max=int(max(lens)),
                   padded_tokens=int(sum(s[0] * s[1] for s in out["batch_shapes"])), real_tokens=int(sum(lens)))
    hits = c("fetch_hits_k5", lambda: [mem.section(s) for s in dense[:5]])
    out["ms"] = c.t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    qs = [json.loads(l)["question"] for l in open(a.questions)][:a.n]

    cold = {"import_torch_transformers_s": IMPORT_S}
    t = time.perf_counter(); models = Models(); mem = Memory(a.memory, models)
    cold["open_memory_s"] = time.perf_counter() - t
    t = time.perf_counter(); models.retriever; cold["load_retriever_s"] = time.perf_counter() - t
    t = time.perf_counter(); models.reranker; cold["load_reranker_s"] = time.perf_counter() - t
    dev = models.retriever.device
    t = time.perf_counter(); one_query(mem, qs[0], "rerank", a.batch_size); cold["first_query_s"] = time.perf_counter() - t

    runs = []
    for q in qs:
        sync(dev); t = time.perf_counter()
        r = one_query(mem, q, "rerank", a.batch_size)
        r["total_ms"] = (time.perf_counter() - t) * 1000
        runs.append(r)

    print(f"device {dev}, dtype {next(models.reranker.model.parameters()).dtype}, torch {torch.__version__}, "
          f"sections {len(mem.section_ids)}, queries {len(qs)}, rerank batch {a.batch_size}")
    print("\ncold start")
    for k, v in cold.items():
        print(f"  {k:32s} {v:7.2f} s")
    stages = list(dict.fromkeys(k for r in runs for k in r["ms"]))
    print("\nwarm, per query (mode=rerank)            median ms     p90 ms   share")
    tot = statistics.median(r["total_ms"] for r in runs)
    for k in stages:
        v = sorted(r["ms"].get(k, 0) for r in runs)
        print(f"  {k:36s} {statistics.median(v):9.1f} {v[int(0.9 * (len(v) - 1))]:10.1f}   {statistics.median(v) / tot:5.1%}")
    print(f"  {'TOTAL':36s} {tot:9.1f}")
    print(f"\npool {statistics.median(r['pool'] for r in runs)} pairs; pair tokens mean "
          f"{statistics.mean(r['pair_tokens_mean'] for r in runs):.0f}, max {max(r['pair_tokens_max'] for r in runs)}; "
          f"padding overhead {statistics.mean(r['padded_tokens'] / r['real_tokens'] - 1 for r in runs):.0%}; "
          f"query tokens {statistics.median(r['tokens_query'] for r in runs)}")
    if a.out:
        Path(a.out).write_text(json.dumps({"cold": cold, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
