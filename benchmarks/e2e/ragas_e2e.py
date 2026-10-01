"""End-to-end answers from each system's context, scored with RAGAS NVIDIA metrics, plus a code-naming check.

Same answerer (Claude Sonnet) and judge (Claude Haiku) for every system. One answer per (question, system).
code_named = the answer names at least one gold symbol (the paired code half of the question).

    pip install "ragas==0.4.3" "langchain-community>=0.3,<0.3.28"; python benchmarks/e2e/ragas_e2e.py"""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from llm import NV, claude, claude_llm  # noqa: E402

HERE = Path.home() / "e2e-bench"
CACHE = HERE / "ragas_cache.json"
PROMPT = """Answer the developer's question using only the context below (documentation excerpts and/or a code graph
of the project). If the context does not contain the answer, say "The context provided does not answer this."
Be concise (at most 4 sentences). Then, on a last line starting with "Code:", name the function(s) or class(es)
that implement this behaviour, if the context shows them (else "Code: unknown").

{ctx}

Question: {q}
Answer:"""


def boot(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import AnswerAccuracy, ContextRelevance, ResponseGroundedness
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "questions.jsonl")}
    ctxs = [json.loads(l) for l in open(HERE / "contexts.jsonl")]
    systems = list(ctxs[0]["ctx"])
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = [(r["id"], s) for r in ctxs for s in systems if f"{r['id']}|{s}" not in cache]
    ctx_of = {(r["id"], s): r["ctx"][s] for r in ctxs for s in systems}

    def answer(job):
        qid, s = job
        return job, claude(PROMPT.format(ctx=ctx_of[job], q=qs[qid]["question"]), "sonnet")

    with ThreadPoolExecutor(8) as ex:
        for n, ((qid, s), resp) in enumerate(ex.map(answer, jobs), 1):
            cache[f"{qid}|{s}"] = {"response": resp}
            if n % 20 == 0 or n == len(jobs):
                CACHE.write_text(json.dumps(cache)); print(f"\ranswered {n}/{len(jobs)}", end="", flush=True)
    print()
    llm = claude_llm("haiku")
    todo = [k for k, v in cache.items() if v.get("response") and any(v.get(m) is None for m in NV)]
    for b in range(0, len(todo), 40):
        chunk = todo[b:b + 40]
        rows = []
        for k in chunk:
            qid, s = k.split("|", 1)
            rows.append({"user_input": qs[qid]["question"], "reference": qs[qid]["answer"],
                         "retrieved_contexts": [ctx_of[(qid, s)]], "response": cache[k]["response"]})
        res = evaluate(EvaluationDataset.from_list(rows), show_progress=False,
                       metrics=[AnswerAccuracy(llm=llm), ContextRelevance(llm=llm), ResponseGroundedness(llm=llm)],
                       run_config=RunConfig(max_workers=8, timeout=900))
        df = res.to_pandas()
        tuples = Counter()
        for k, (_, row) in zip(chunk, df.iterrows()):
            vals = [row.get(m) for m in NV]
            for m, v in zip(NV, vals):
                cache[k][m] = None if v is None or v != v else float(v)
            tuples[tuple(cache[k][m] for m in NV)] += 1
        top, cnt = tuples.most_common(1)[0]
        if cnt > len(chunk) // 3 and top in ((0.0, 0.5, 0.5), (0.0, 0.0, 0.0)):   # silent judge failures
            print(f"  chunk {b}: suspicious tuple {top} x{cnt}; clearing for re-judge")
            for k in chunk:
                if tuple(cache[k][m] for m in NV) == top:
                    for m in NV:
                        cache[k].pop(m, None)
        CACHE.write_text(json.dumps(cache))
        print(f"judged {min(b + 40, len(todo))}/{len(todo)}", flush=True)
    # report
    table = {}
    for s in systems:
        per = {m: [] for m in NV + ["nv_mean", "code_named"]}
        for qid, q in qs.items():
            v = cache.get(f"{qid}|{s}", {})
            if any(v.get(m) is None for m in NV):
                continue
            for m in NV:
                per[m].append(v[m])
            per["nv_mean"].append(float(np.mean([v[m] for m in NV])))
            code_line = (v["response"] or "").split("Code:")[-1]
            names = [g["label"].strip(".").replace("()", "").split(".")[-1] for g in q["gold_code"]]
            per["code_named"].append(float(any(re.search(rf"\b{re.escape(n)}\b", code_line) for n in names)))
        table[s] = {m: boot(v) for m, v in per.items()} | {"n": len(per["nv_mean"])}
    base = "graphify-full" if "graphify-full" in systems else "graphify-ast"
    paired = {}
    for s in systems:
        if s == base:
            continue
        d = []
        for qid in qs:
            a, b = cache.get(f"{qid}|{base}", {}), cache.get(f"{qid}|{s}", {})
            if all(a.get(m) is not None for m in NV) and all(b.get(m) is not None for m in NV):
                d.append(np.mean([b[m] for m in NV]) - np.mean([a[m] for m in NV]))
        paired[s] = boot(d)
    (HERE / "ragas_report.json").write_text(json.dumps({"table": table, f"nv_mean_vs_{base}": paired}, indent=1))
    for s, t in table.items():
        print(f"{s:14s} n={t['n']:3d} " + "  ".join(f"{m} {t[m][0]:.3f}" for m in NV + ["nv_mean", "code_named"]))
    for s, (d, lo, hi) in paired.items():
        print(f"  {s} vs {base}: nv_mean {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
