"""How noisy are the grounding flags? Counts, per search hit, how often `! doc may be stale`, `! doc/code conflict`
and `verify in:` fire. Model-free: hits are the BM25 top-k (no GPU), which is enough to measure the flags.

    DEFROST_HOME=<home> python scripts/grounding_audit.py --memory <out dir> --queries q.txt [--k 5] [--out flags.json]
--queries: one query per line, or a questions.jsonl with a "question" field."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from defrost_ai.memory import Memory
from defrost_ai.retrieval import keyword


def load_queries(path: str, n: int) -> list[str]:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    if path.endswith(".jsonl"):
        lines = [json.loads(l)["question"] for l in lines]
    return lines[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    mem = Memory(a.memory)
    rows = []
    for q in load_queries(a.queries, a.n):
        for sid, _ in keyword.bm25_sections(mem.db, q, a.k):
            s = mem.section(sid)
            verify, stale = mem.grounding(sid, s["path"])
            missing = mem.missing_names(sid, s["path"])
            rows.append({"q": q, "sid": sid, "path": s["path"], "lines": [s["line_start"], s["line_end"]],
                         "verify": verify, "stale": stale, "missing": missing})
    n = len(rows)
    share = lambda f: sum(1 for r in rows if f(r)) / n if n else 0.0
    print(f"{a.memory}: {n} hits")
    print(f"  hits with '! doc may be stale'  {share(lambda r: r['stale']):6.1%}   "
          f"(flags total {sum(len(r['stale']) for r in rows)})")
    print(f"  hits with '! doc/code conflict' {share(lambda r: r['missing']):6.1%}   "
          f"(names total {sum(len(r['missing']) for r in rows)})")
    print(f"  hits with 'verify in:'          {share(lambda r: r['verify']):6.1%}   "
          f"(files per hit: mean {statistics.mean(len(r['verify']) for r in rows):.2f}, "
          f"max {max(len(r['verify']) for r in rows)})")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
