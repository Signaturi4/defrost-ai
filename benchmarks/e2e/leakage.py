"""13-gram overlap between the e2e eval repos (docs + python source) and every file our models were trained on."""
import json, re, sys
from pathlib import Path
REPOS = Path.home() / "e2e-bench/repos"
TRAIN = [Path(__file__).resolve().parents[2] / "data/training" / p for p in
         ["corpus/corpus_mntp.txt", "corpus/corpus_cgsa.txt", "retriever/train.jsonl", "reranker/rerank_train.jsonl"]] + \
        []                          # + the Defrost-Rerank v2 candidate groups (not distributed)
def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + 13]) for i in range(len(w) - 12)}
ev = {}
for repo in sorted(REPOS.iterdir()):
    for f in list(repo.rglob("*.md")) + list(repo.rglob("*.rst")) + list(repo.rglob("*.py")):
        if ".git" in f.parts: continue
        for g in grams(f.read_text(errors="ignore")):
            ev.setdefault(g, set()).add(repo.name)
print("eval 13-grams", len(ev), flush=True)
hits = {}
for f in TRAIN:
    n = 0
    with open(f, errors="ignore") as fh:
        for line in fh:
            for g in grams(line):
                if g in ev:
                    for r in ev[g]:
                        hits.setdefault((r, f.name), set()).add(g)
            n += 1
    print(f.name, "scanned", n, flush=True)
per_repo = {}
for (r, fn), gs in sorted(hits.items()):
    print(r, fn, len(gs), list(gs)[:3])
tot = {r: sum(1 for g, rs in ev.items() if r in rs) for r in {x for s in ev.values() for x in s}}
print("eval grams per repo", tot)
json.dump(sorted({g for gs in hits.values() for g in gs}), open(Path.home() / "e2e-bench/leaked_grams.json", "w"))
