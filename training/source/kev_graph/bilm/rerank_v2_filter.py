"""Kev-Rerank v2, step 2: drop likely false negatives among the new hard negatives (siblings, changelog sections,
same-book prose). A candidate is dropped when its Kev-Ret-B cosine to the query is >= FILTER_MARGIN x the positive's
(NV-Retriever's TopK-PercPos rule): a sibling that scores as high as the gold section probably answers the question too.

Runs in the kev-memory venv (it uses the packaged Kev-Ret-B):
    ../kev-memory/.venv/bin/python -m kev_graph.bilm.rerank_v2_filter
reads kev_graph/data/rerank_v2/candidates.jsonl -> candidates_keep.jsonl ({"keep": [bool per candidate]})."""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("kev_graph/data/rerank_v2")
FILTER_MARGIN = 0.95


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "kev-memory"))
    from kev_memory.models.encoder import KevRetriever
    from kev_memory.models.weights import RETRIEVAL_INSTRUCTION
    groups = [json.loads(l) for l in open(OUT / "candidates.jsonl")]
    queries = list(dict.fromkeys(g["query"] for g in groups))
    docs = list(dict.fromkeys(t for g in groups for t in [g["positive"]] + [c[0] for c in g["cand"]]))
    print(f"{len(groups)} groups: {len(queries)} queries, {len(docs)} unique texts", flush=True)
    ret = KevRetriever(max_tokens=256)          # a cosine threshold needs no full-length encoding
    t0 = time.time()
    qv = ret._encode(queries, 32, prefix=f"{RETRIEVAL_INSTRUCTION}: ")
    print(f"queries in {time.time() - t0:.0f}s", flush=True)
    dv = np.zeros((len(docs), qv.shape[1]), np.float32)
    step = 2000
    for b in range(0, len(docs), step):
        dv[b:b + step] = ret.embed_documents(docs[b:b + step], batch_size=32)
        print(f"  docs {min(b + step, len(docs))}/{len(docs)} | {time.time() - t0:.0f}s", flush=True)
    qi = {q: i for i, q in enumerate(queries)}
    di = {d: i for i, d in enumerate(docs)}
    n_drop = n_all = 0
    with open(OUT / "candidates_keep.jsonl", "w") as f:
        for g in groups:
            q = qv[qi[g["query"]]]
            pos = float(dv[di[g["positive"]]] @ q)
            keep = [bool(float(dv[di[c[0]]] @ q) < FILTER_MARGIN * pos) for c in g["cand"]]
            n_drop += keep.count(False); n_all += len(keep)
            f.write(json.dumps({"keep": keep, "cos_pos": round(pos, 4)}) + "\n")
    print(f"dropped {n_drop}/{n_all} candidates ({n_drop / max(1, n_all):.1%}) as likely false negatives")


if __name__ == "__main__":
    main()
