"""Parity check: does this package reproduce the evaluated system exactly?

Compares a memory built by kev-memory with the research build (kev repo) on the same sources:
  1. sections: ids, heading paths, line ranges, text
  2. vectors: Kev-Ret-B section vectors (max |diff|, min cosine)
  3. rankings: for each question, BM25 top-50, dense top-50 and the reranker order, against the cached research run
  4. metrics: nDCG@10 per mode on the suite split

    python scripts/check_parity.py --memory ~/.kev-memory/parity-heldout \\
        --research-db ~/heldout-memory/text_kb.sqlite \\
        --research-vectors ~/heldout-memory/vectors-kevret-sup-B-final-cgsa.npz \\
        --research-rows /path/kev/runs/kev_graph/router/heldout__kevret_runs_kev_graph_sup-B_final+cgsa.json \\
        --suite benchmarks/heldout/questions.jsonl --split dev"""
import argparse
import json
import sqlite3

import numpy as np

from kev_memory.evaluation.benchmark import run as benchmark
from kev_memory.memory import Memory, Models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--research-db", required=True)
    ap.add_argument("--research-vectors", required=True)
    ap.add_argument("--research-rows", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-rank", type=int, default=0, help="questions to compare rankings on (0 = all)")
    a = ap.parse_args()
    models = Models()
    mem = Memory(a.memory, models)
    ok = True

    old = sqlite3.connect(a.research_db)
    cols = "id, heading_path, line_start, line_end, text"
    new_rows = mem.db.execute(f"SELECT {cols} FROM sections ORDER BY rowid").fetchall()
    old_rows = old.execute(f"SELECT {cols} FROM sections ORDER BY rowid").fetchall()
    same = new_rows == old_rows
    ok &= same
    print(f"[1] sections: new {len(new_rows)} old {len(old_rows)} identical={same}")
    if not same:
        diff = [(n, o) for n, o in zip(new_rows, old_rows) if n != o][:3]
        print("    first differences:", diff)

    ov = np.load(a.research_vectors)
    old_vec = dict(zip(ov["section_ids"], ov["section_vecs"].astype(np.float32)))
    common = [i for i, s in enumerate(mem.section_ids) if s in old_vec]
    A = mem.section_vecs[common]
    B = np.stack([old_vec[mem.section_ids[i]] for i in common])
    max_abs, min_cos = float(np.abs(A - B).max()), float((A * B).sum(1).min())
    ok &= min_cos > 0.999
    print(f"[2] vectors: {len(common)} compared, max|diff| {max_abs:.2e}, min cosine {min_cos:.6f}")

    rows = json.loads(open(a.research_rows).read())["rows"]
    rows = rows[:a.n_rank] if a.n_rank else rows
    agree = {"bm25": 0, "dense": 0, "rerank": 0}
    for r in rows:
        bm25, dense, _ = mem.candidates(r["question"])
        agree["bm25"] += bm25 == [s for s, _ in r["bm25"]]
        agree["dense"] += dense[:10] == [s for s, _ in r["dense"]][:10]
        scores = mem.rerank_scores(r["question"], bm25, dense)
        new_order = sorted(scores, key=lambda s: -scores[s])[:10]
        old_order = [s for s, _ in sorted(r["rerank"], key=lambda x: -x[1])][:10]
        agree["rerank"] += new_order == old_order
    print(f"[3] rankings identical on {len(rows)} questions: " + ", ".join(f"{k} {v}/{len(rows)}" for k, v in agree.items()))
    ok &= all(v == len(rows) for v in agree.values())

    res = benchmark(a.suite, a.memory, a.split, models)
    print("[4] nDCG@10 " + " ".join(f"{m} {t['ndcg@10']:.3f}" for m, t in res["table"].items()))
    print("PARITY OK" if ok else "PARITY DIFFERENCES (see above)")


if __name__ == "__main__":
    main()
