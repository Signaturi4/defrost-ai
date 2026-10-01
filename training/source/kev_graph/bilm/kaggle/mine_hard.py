"""E1: dense hard negatives mined by Kev-Ret-B (ANCE-style), per source, on a Kaggle GPU.

For each training row: encode the query (its source instruction, not pooled) and every positive of the same source,
take the top-30, drop the row's own positive, same-section chunks (techdoc: same path + heading) and near-copies
(5-shingle Jaccard > 0.5 with the positive), then sample one from ranks 3-30 (the top 2 are the likeliest false
negatives). The row keeps its BM25 negative as `negative_bm25` and gets the dense one as `negative`.

    python mine_hard.py --data DIR --adapter DIR --mntp DIR --cgsa DIR --out DIR/train_e1.jsonl"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kev_graph.bilm.kaggle.eval_public_sup import KevRet  # noqa: E402


def shingles(text, k=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train_v2.jsonl")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mntp", required=True)
    ap.add_argument("--cgsa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_doc", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(Path(args.data) / "train_v2.jsonl")]
    enc = KevRet(args.mntp, args.cgsa, args.adapter, torch.device("cuda"), max_tokens=args.max_doc)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["source"]].append(i)
    stats = defaultdict(int)
    for src, idx in by.items():
        t0 = time.time()
        pool = list(dict.fromkeys(rows[i]["positive"] for i in idx))
        pos_ix = {p: k for k, p in enumerate(pool)}
        V = torch.tensor(enc.encode(pool, bs=48), device="cuda", dtype=torch.float16)
        ins = rows[idx[0]]["instruction"]
        Q = torch.tensor(enc.encode([rows[i]["query"] for i in idx], prefix=f"{ins}: ", bs=64), device="cuda",
                         dtype=torch.float16)
        for b in range(0, len(idx), 1024):
            top = torch.topk(Q[b:b + 1024] @ V.T, k=min(30, len(pool)), dim=1).indices.cpu().numpy()
            for j, cand in enumerate(top):
                r = rows[idx[b + j]]
                me = pos_ix[r["positive"]]
                ps = shingles(r["positive"])
                ok = []
                for rank, c in enumerate(cand):
                    if c == me or rank < 2:
                        continue
                    t = pool[c]
                    sj = shingles(t)
                    if len(ps & sj) / max(1, len(ps | sj)) > 0.5:
                        continue
                    ok.append(t)
                r["negative_bm25"] = r["negative"]
                if ok:
                    r["negative"] = rng.choice(ok[:10])
                    stats[f"{src}_dense"] += 1
                else:
                    stats[f"{src}_kept_bm25"] += 1
        print(f"{src}: {len(idx)} rows, pool {len(pool)}, {time.time() - t0:.0f}s", flush=True)
        del V, Q
        torch.cuda.empty_cache()
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(dict(stats))


if __name__ == "__main__":
    main()
