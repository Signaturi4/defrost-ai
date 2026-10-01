"""Search modes.

Two modes for users:
  accurate  (default) hybrid when BM25 and the dense retriever agree on the top section, otherwise Defrost-Rerank
            reorders the top 40. Best quality: locked-test nDCG@10 0.838 (always reranking: 0.839) while the reranker
            runs on about half the queries. Slower: ~1.2-1.8 s when the reranker runs (Apple M5, MLX), ~0.1 s when not.
  fast      the best ranking without the reranker: reciprocal-rank fusion of BM25 and Defrost-Ret-B (hybrid).
            ~0.05-0.1 s per query; locked-test nDCG@10 0.776 (dense alone 0.749, BM25 alone 0.696).

Expert modes (benchmarks, debugging): bm25, dense, hybrid, rerank (always rerank), all (RRF of the three).
`fast` meant the reranking policy before 1.2; that policy is now called `accurate`."""
from __future__ import annotations

from collections import defaultdict

RRF_K = 60
POOL = 20
DEPTH = 50
MODES = ("bm25", "dense", "hybrid", "rerank", "all", "accurate")      # internal policies
USER_MODES = {"accurate": "accurate", "fast": "hybrid"}                 # what users choose between
DEFAULT_MODE = "accurate"


def normalize(mode: str | None) -> str:
    """User or expert mode name -> internal policy. None -> the configured default (accurate)."""
    mode = (mode or DEFAULT_MODE).lower()
    if mode in USER_MODES:
        return USER_MODES[mode]
    if mode in MODES:
        return mode
    raise ValueError(f"unknown search mode {mode!r}: use 'accurate' (default) or 'fast'")


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
    if mode == "accurate":
        return not top_agrees(bm25, dense)
    return mode in ("rerank", "all")


def top_agrees(bm25: list[str], dense: list[str]) -> bool:
    return bool(bm25) and bool(dense) and bm25[0] == dense[0]


def rank(mode: str, bm25: list[str], dense: list[str], rerank_scores: dict[str, float] | None = None):
    """-> (ranked section ids, mode actually used)"""
    if mode == "accurate":
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


# Adaptive k (k="auto"): send the smallest k whose calibrated Defrost-Ret-B probability mass reaches AUTO_MASS.
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
