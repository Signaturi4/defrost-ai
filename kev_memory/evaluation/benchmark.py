"""Benchmark a built memory on a frozen question suite: every retrieval mode, nDCG@10 / Recall@1 / Recall@10 / MRR,
paired bootstrap vs BM25. The reranker is run once per question on the full pool, so every mode is scored on the
same candidates (and `fast` is computed exactly as served).

    kev-memory benchmark --suite benchmarks/heldout/questions.jsonl --memory ~/.kev-memory/heldout --split dev

Suite rows: {"id", "split", "question", "answer", "gold": [{"path", "lines": [a, b]}], ...}.
Multi-hop rows also carry "gold_groups": [[span, ...], [span, ...]] (one group per evidence); they add group
recall@5/@10 (share of groups found) and all-evidence@10 (every group found). `only` keeps rows whose "suite" field
matches (a suite file that spans several memories, e.g. benchmarks/multihop)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev_memory.evaluation.metrics import first_hit, group_hits, ndcg_at_10, overlaps, paired_bootstrap
from kev_memory.memory import Memory, Models
from kev_memory.retrieval import policy

MODES = ["bm25", "dense", "hybrid", "rerank", "all", "fast"]


def run(suite: str | Path, memory_dir: str | Path, split: str = "dev", models: Models | None = None, log=None,
        save_rankings: str | Path | None = None, only: str | None = None) -> dict:
    items = [json.loads(l) for l in open(suite)]
    if split != "all":
        items = [it for it in items if it.get("split", split) == split]
    if only:
        items = [it for it in items if it.get("suite") == only]
    mem = Memory(memory_dir, models or Models())
    meta = {s: mem.section(s) for s in mem.section_ids}
    per = defaultdict(lambda: defaultdict(list))
    rankings = []
    for n, it in enumerate(items, 1):
        q = it["question"]
        bm25, dense, _ = mem.candidates(q)
        scores = mem.rerank_scores(q, bm25, dense)
        relevant = {s for s in set(bm25) | set(dense) if overlaps(meta[s]["path"], meta[s]["line_start"],
                                                                   meta[s]["line_end"], it["gold"])}
        groups = [{s for s in set(bm25) | set(dense) if overlaps(meta[s]["path"], meta[s]["line_start"],
                                                                 meta[s]["line_end"], g)} for g in it.get("gold_groups", [])]
        row = {"id": it["id"], "used": {}}
        for m in MODES:
            ranked, used = policy.rank(m, bm25, dense, scores)
            r = first_hit(ranked, relevant)
            per[m]["ndcg@10"].append(ndcg_at_10(r))
            per[m]["recall@1"].append(float(r == 0))
            per[m]["recall@10"].append(float(r is not None and r < 10))
            per[m]["mrr@10"].append(0.0 if r is None or r >= 10 else 1 / (r + 1))
            if len(groups) > 1:
                per[m]["group_recall@5"].append(group_hits(ranked, groups, 5) / len(groups))
                per[m]["group_recall@10"].append(group_hits(ranked, groups, 10) / len(groups))
                per[m]["all_evidence@10"].append(float(group_hits(ranked, groups, 10) == len(groups)))
            row["used"][m] = used
            row[m] = ranked[:10]
        rankings.append(row)
        if log:
            log(f"{n}/{len(items)}")
    table = {m: {k: float(np.mean(v)) for k, v in per[m].items()} for m in MODES}
    vs = {m: dict(zip(["delta", "ci_low", "ci_high", "p_le_0"],
                      paired_bootstrap(per["bm25"]["ndcg@10"], per[m]["ndcg@10"]))) for m in MODES if m != "bm25"}
    vs_fast = {}
    if per["fast"].get("all_evidence@10"):
        vs_fast = {m: dict(zip(["delta", "ci_low", "ci_high", "p_le_0"],
                               paired_bootstrap(per["fast"]["all_evidence@10"], per[m]["all_evidence@10"])))
                   for m in MODES if m != "fast"}
    rerank_share = float(np.mean([r["used"]["fast"] == "rerank" for r in rankings])) if rankings else 0.0
    if save_rankings:
        Path(save_rankings).write_text("\n".join(json.dumps(r) for r in rankings))
    return {"suite": str(suite), "split": split, "n": len(items), "table": table, "vs_bm25_ndcg": vs,
            "vs_fast_all_evidence": vs_fast,
            "fast_rerank_share": rerank_share, "per_question": {m: dict(v) for m, v in per.items()}}


def print_report(res: dict):
    print(f"{res['suite']} [{res['split']}] n={res['n']}; fast used the reranker on {res['fast_rerank_share']:.0%}")
    print(f"{'mode':8s} {'nDCG@10':>8s} {'R@1':>6s} {'R@10':>6s} {'MRR@10':>7s}   nDCG@10 vs bm25 [95% CI]")
    for m, t in res["table"].items():
        v = res["vs_bm25_ndcg"].get(m)
        ci = f"   {v['delta']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]" if v else ""
        print(f"{m:8s} {t['ndcg@10']:8.3f} {t['recall@1']:6.3f} {t['recall@10']:6.3f} {t['mrr@10']:7.3f}{ci}")
    if res.get("vs_fast_all_evidence"):
        print(f"{'mode':8s} {'grp@5':>6s} {'grp@10':>6s} {'all@10':>6s}   all-evidence@10 vs fast [95% CI]")
        for m, t in res["table"].items():
            v = res["vs_fast_all_evidence"].get(m)
            ci = f"   {v['delta']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]" if v else ""
            print(f"{m:8s} {t['group_recall@5']:6.3f} {t['group_recall@10']:6.3f} {t['all_evidence@10']:6.3f}{ci}")
