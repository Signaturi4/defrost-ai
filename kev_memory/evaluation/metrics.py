"""Retrieval metrics used by the benchmark (same definitions as the published results).

gold match: a retrieved section is relevant when it overlaps a gold span by >= 50% of either span's lines
nDCG@10:    single-relevant-item form, 1/log2(rank+2) for the first relevant hit in the top 10
bootstrap:  paired, over questions, 5000 resamples -> mean difference, 95% CI, P(diff <= 0)"""
from __future__ import annotations

import numpy as np


def overlaps(path: str, a: int, b: int, gold: list[dict]) -> bool:
    for g in gold:
        if g["path"] != path:
            continue
        ga, gb = g["lines"]
        inter = min(b, gb) - max(a, ga) + 1
        if inter > 0 and (inter >= 0.5 * (gb - ga + 1) or inter >= 0.5 * (b - a + 1)):
            return True
    return False


def first_hit(ranked: list[str], relevant: set[str], depth: int = 50) -> int | None:
    return next((i for i, s in enumerate(ranked[:depth]) if s in relevant), None)


def ndcg_at_10(rank: int | None) -> float:
    return 0.0 if rank is None or rank >= 10 else float(1 / np.log2(rank + 2))


def paired_bootstrap(baseline, system, n: int = 5000, seed: int = 0):
    d = np.asarray(system, float) - np.asarray(baseline, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs <= 0).mean())


def group_hits(ranked: list[str], groups: list[set[str]], k: int) -> int:
    """Number of evidence groups with at least one relevant section in the top k (multi-hop questions)."""
    top = set(ranked[:k])
    return sum(bool(top & g) for g in groups)
