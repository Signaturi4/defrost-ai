"""Mini health check, run after every optimization phase: does the system still work and compute the same thing?

1. parity: query vectors and reranker scores for 3 fixed questions vs a saved fp32 reference
   (first run with --save-reference writes it)
2. smoke: every search mode returns hits with finite scores, k="auto" works
3. latency: cold load times and warm per-query time (rerank mode)
4. service (optional, --service): the resident HTTP service answers /health and a search

    DEFROST_MODELS=<weights> python scripts/health_check.py [--save-reference] [--service]
Exit code 1 when a check fails. Latency is reported, never failed on."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/private/latency/health_reference.json"
MEMORY = Path("~/.defrost-ai/parity-heldout").expanduser()
if not MEMORY.exists():                                            # home from before the rename
    MEMORY = Path("~/.kev-memory/parity-heldout").expanduser()
QUESTIONS = ROOT / "benchmarks/heldout/questions.jsonl"
MODES = ("accurate", "fast", "rerank", "hybrid", "dense", "bm25", "all")


def kendall_tau(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2) if n > 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-reference", action="store_true")
    ap.add_argument("--service", action="store_true")
    ap.add_argument("--score-tol", type=float, default=0.05, help="max |score diff| allowed vs the fp32 reference")
    a = ap.parse_args()
    fails = []

    t = time.perf_counter()
    from defrost_ai.memory import Memory, Models
    from defrost_ai.retrieval import policy
    import_s = time.perf_counter() - t
    qs = [json.loads(l)["question"] for l in open(QUESTIONS)][:3]
    t = time.perf_counter(); models = Models(); mem = Memory(MEMORY, models)
    _ = models.retriever; load_ret = time.perf_counter() - t
    t = time.perf_counter(); _ = models.reranker; load_rr = time.perf_counter() - t

    # 1. parity
    t = time.perf_counter()
    cur = []
    for q in qs:
        bm25, dense, _ = mem.candidates(q)
        cur.append({"q": q, "qvec": mem.models.retriever.embed_query(q).tolist(),
                    "scores": mem.rerank_scores(q, bm25, dense)})
    first_s = time.perf_counter() - t
    if a.save_reference:
        REF.parent.mkdir(parents=True, exist_ok=True)
        REF.write_text(json.dumps(cur))
        print(f"reference saved: {REF}")
    ref = json.loads(REF.read_text()) if REF.exists() else None
    if ref:
        for r, c in zip(ref, cur):
            cos = float(np.dot(r["qvec"], c["qvec"]) / (np.linalg.norm(r["qvec"]) * np.linalg.norm(c["qvec"])))
            ids = [s for s in r["scores"] if s in c["scores"]]
            ra, ca = [r["scores"][s] for s in ids], [c["scores"][s] for s in ids]
            diff = max(abs(x - y) for x, y in zip(ra, ca))
            top = max(ids, key=lambda s: r["scores"][s]) == max(ids, key=lambda s: c["scores"][s])
            tau = kendall_tau(ra, ca)
            ok = cos > 0.999 and diff <= a.score_tol and top and len(ids) == len(r["scores"])
            print(f"parity  qvec cos {cos:.6f}  score max|diff| {diff:.4f}  top1 {'same' if top else 'DIFF'}  "
                  f"tau {tau:.3f}  pool {len(ids)}/{len(r['scores'])}  {'ok' if ok else 'FAIL'}")
            if not ok:
                fails.append(f"parity: {c['q'][:50]}")
    else:
        print("parity  no reference yet (run with --save-reference on the fp32 baseline)")

    # 2. smoke
    for mode in MODES:
        for k in (5, "auto"):
            res = mem.search(qs[0], mode=mode, k=k)
            bad = [h for h in res["hits"] if h["rerank_score"] is not None and not math.isfinite(h["rerank_score"])]
            if not res["hits"] or bad:
                fails.append(f"smoke: mode {mode} k {k}")
    print(f"smoke   {len(MODES)} modes x k in (5, auto): {'ok' if not any(f.startswith('smoke') for f in fails) else 'FAIL'}")

    # 3. latency
    getattr(models.reranker, "_cache", {}).clear()                   # uncached: the model really runs
    times = []
    for q in qs:
        t = time.perf_counter(); mem.search(q, mode="rerank", k=5); times.append(time.perf_counter() - t)
    cached = []
    for q in qs:
        t = time.perf_counter(); mem.search(q, mode="rerank", k=5); cached.append(time.perf_counter() - t)
    backend = getattr(models.reranker, "backend", "torch")
    print(f"latency [{backend}] import {import_s:.1f}s  load retriever {load_ret:.1f}s  reranker {load_rr:.1f}s  "
          f"first 3 queries {first_s:.1f}s  warm rerank median {statistics.median(times) * 1000:.0f} ms "
          f"(max {max(times) * 1000:.0f})  repeated query {statistics.median(cached) * 1000:.0f} ms")

    # 4. service
    if a.service:
        from defrost_ai.service import client
        client.ensure_service()
        h = client._call("GET", "/health")
        t = time.perf_counter(); r = client.search(qs[1], None, "accurate", "auto"); dt = time.perf_counter() - t
        ok = h.get("ok") and r.get("hits")
        print(f"service build {h.get('build')}  search {dt * 1000:.0f} ms, {len(r.get('hits', []))} hits  "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("service")

    print("HEALTH", "OK" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
