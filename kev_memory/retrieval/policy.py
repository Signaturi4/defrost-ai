"""Retrieval modes and the default `fast` policy.

Given the BM25 list, the dense list and (when computed) reranker scores for one query:
  bm25     keyword ranking
  dense    Kev-Ret-B cosine ranking
  hybrid   reciprocal-rank fusion (k=60) of bm25 and dense
  rerank   Kev-Rerank order of the pool (BM25 top-20 + dense top-20), the rest in hybrid order
  all      RRF of bm25, dense and rerank (equal weights)
  fast     hybrid when BM25 and dense agree on the top section, otherwise rerank  <- default

Why `fast` is the default (docs/EVALUATION.md): on the locked test sets it matches `rerank` (nDCG@10 0.814 vs 0.802)
while calling the reranker for only ~half the queries; it has no fitted parameters, so it cannot overfit. A fitted
decision-tree router did not generalise across domains."""
from __future__ import annotations

from collections import defaultdict

RRF_K = 60
POOL = 20
DEPTH = 50
MODES = ("bm25", "dense", "hybrid", "rerank", "all", "fast")


def rrf(ranked_lists: list[list[str]], weights=None) -> list[str]:
    weights = weights or [1.0] * len(ranked_lists)
    score = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        for r, x in enumerate(lst):
            score[x] += w / (RRF_K + r + 1)
    return sorted(score, key=score.get, reverse=True)


def rerank_pool(bm25: list[str], dense: list[str]) -> list[str]:
    return list(dict.fromkeys(bm25[:POOL] + dense[:POOL]))


def needs_reranker(mode: str, bm25: list[str], dense: list[str]) -> bool:
    if mode == "fast":
        return not top_agrees(bm25, dense)
    return mode in ("rerank", "all")


def top_agrees(bm25: list[str], dense: list[str]) -> bool:
    return bool(bm25) and bool(dense) and bm25[0] == dense[0]


def rank(mode: str, bm25: list[str], dense: list[str], rerank_scores: dict[str, float] | None = None):
    """-> (ranked section ids, mode actually used)"""
    if mode == "fast":
        mode = "hybrid" if top_agrees(bm25, dense) else "rerank"
    if mode == "bm25":
        return bm25, mode
    if mode == "dense":
        return dense, mode
    hybrid = rrf([bm25, dense])
    if mode == "hybrid":
        return hybrid, mode
    reranked = sorted(rerank_scores, key=lambda s: -rerank_scores[s])
    if mode == "rerank":
        seen = set(reranked)
        return reranked + [s for s in hybrid if s not in seen], mode
    if mode == "all":
        return rrf([bm25, dense, reranked]), mode
    raise ValueError(f"unknown mode {mode!r}; one of {MODES}")


# Adaptive k (k="auto"): send the smallest k whose calibrated Kev-Ret-B probability mass reaches AUTO_MASS.
# Probabilities are softmax(cosine * 20 / T) over the query's candidates (BM25 top-50 + dense top-50). T and the mass
# were fitted on dev (3 domains, leave-one-domain-out) and read once on the locked test: -18% context tokens,
# hit@k -0.006 [-0.017, 0] vs k=5 (docs/EVALUATION.md).
AUTO_T = 0.75
AUTO_MASS = 0.94
AUTO_K_MAX = 5
DENSE_SCALE = 20.0


def auto_k(ranked: list[str], cosine: dict[str, float], k_max: int = AUTO_K_MAX, mass: float = AUTO_MASS,
           T: float = AUTO_T) -> int:
    import math
    if not cosine:
        return k_max
    z = {s: c * DENSE_SCALE / T for s, c in cosine.items()}
    m = max(z.values())
    total = sum(math.exp(v - m) for v in z.values())
    acc = 0.0
    for i, s in enumerate(ranked[:k_max], 1):
        acc += math.exp(z[s] - m) / total if s in z else 0.0
        if acc >= mass:
            return i
    return k_max
