# Evaluation

## Question

Does the shipped stack retrieve better documentation than a plain SQLite FTS5 (BM25) index of the same heading
sections? That index is the baseline "Text KB" of the graphify-based memory.

## Protocol

- **Suites.** Each suite is frozen, and its sha256 is recorded below. Every question carries gold spans
  (path + line range). A retrieved section counts as relevant when it overlaps a gold span by ≥ 50%.
- **Splits.** dev and test are split by hash. Every choice was made on dev: model, checkpoint, pool size and policy.
  The test split was scored once, after the policy was frozen and written down.
- **Leakage.** A 13-gram gate compares all training data against every suite.
- **Statistics.** Paired bootstrap over questions, 5000 resamples. Differences are reported with 95% CIs.
- **RAGAS.** RAGAS 0.4.3 NVIDIA metrics (answer accuracy, context relevance, response groundedness).
  Claude Sonnet answers from the top-3 sections and Claude Haiku judges. Every run is checked for silent judge
  failures, which show up as a single score tuple dominating a batch.

| suite | questions | domain | written by |
|---|---|---|---|
| held-out (`benchmarks/heldout`, sha 78c2c2b8…) | 150 (dev 44 / test 106) | jinja, werkzeug, marshmallow docs + code | LLM-generated, LLM-verified |
| private product repos | 98 (dev 29 / test 69) | a 7-repo product (backend, web, mobile, infra, recsys) | LLM-generated, LLM-verified; not distributed |
| books (`benchmarks/books`, sha 3b16271f…) | 39 (7 multi-hop) | Pro Git, Eloquent JavaScript, 500 Lines or Less | hand-written, evidence-quoted |
| ODEX / DS-1000 (CodeRAG-Bench) | 201 / 513 | Python library documentation (34k docs) | public |

## Locked test results (scored once)

nDCG@10:

| test set | BM25 | dense | hybrid | rerank | fast (default) |
|---|---|---|---|---|---|
| held-out (106) | 0.703 | 0.803 | 0.799 | 0.800 → **0.846** | 0.814 → **0.840** |
| private repos (69) | 0.684 | 0.667 | 0.740 | 0.804 → **0.829** | 0.814 → **0.835** |

Arrows show Defrost-Rerank v1 (weights v1.0.0) → v2 (weights v1.1.0, the default). BM25, dense and hybrid do not use the
reranker. The claims below were pre-registered and tested with v1.

Pre-registered claim, stack vs BM25:
- held-out: +0.097 [+0.038, +0.158]
- private repos: +0.120 [+0.044, +0.197]

Both are **PASS**. `fast` vs BM25 is +0.111 and +0.130; both CIs are above zero.

RAGAS NVIDIA, mean of the three metrics, all 175 test questions:

| | BM25 | dense | hybrid | rerank | fast |
|---|---|---|---|---|---|
| nv_mean | 0.791 | 0.842 | 0.855 | **0.891** | 0.879 |
| answer accuracy | 0.689 | 0.736 | 0.746 | **0.787** | 0.771 |
| context relevance | 0.891 | 0.937 | 0.941 | **0.971** | 0.964 |
| groundedness | 0.794 | 0.854 | 0.877 | **0.916** | 0.901 |

- rerank vs BM25: +0.100 [+0.054, +0.148]
- fast vs BM25: +0.088 [+0.042, +0.134]
- fast vs rerank: −0.012 [−0.032, +0.005]. fast sends 46–57% of queries to the reranker.

RAGAS was run with Defrost-Rerank v1 and has not been re-run for v2 (it costs Claude calls).

### Defrost-Rerank v2 vs v1 (locked test, read once on 2026-10-01)

v2 continues v1's training (800 steps, lr 3e-5) on 22.5k examples whose negatives add up to two same-file sibling
sections and two changelog / release-note sections per question (19k sibling and 17k changelog negatives in
total), plus 932 long-prose questions. Both rerankers score the same candidate pools, so the comparison is paired.

| test set | n | `rerank` v1 → v2 | `fast` v1 → v2 | `all` v1 → v2 |
|---|---|---|---|---|
| held-out | 106 | 0.800 → 0.846, +0.046 [+0.021, +0.073] | 0.814 → 0.840, +0.027 [+0.006, +0.050] | 0.851 → 0.854 |
| private repos | 69 | 0.804 → 0.829, +0.025 [−0.001, +0.054] | 0.814 → 0.835, +0.021 [+0.004, +0.042] | 0.794 → 0.789 |
| pooled | 175 | **+0.037 [+0.019, +0.056]** | **+0.024 [+0.010, +0.041]** | −0.001 [−0.011, +0.010] |

On dev, v2 gained on every suite: held-out 0.870 → 0.890, private 0.828 → 0.832, books 0.862 → 0.902 and a new
long-prose suite (47 questions) 0.772 → 0.805 (`rerank`).

## Dev and other suites (used for choices)

| system | held-out dev | private dev | books | ODEX | DS-1000 |
|---|---|---|---|---|---|
| BM25 | 0.711 | 0.678 | 0.804 | 0.134 | 0.135 |
| bge-small-en-v1.5 | – | 0.515 | 0.829 | 0.196 | 0.287 |
| bge-base-en-v1.5 | 0.799 | – | – | 0.222 | 0.273 |
| Defrost-Ret-B dense | **0.886** | 0.541 | 0.855 | **0.230** | 0.257 |
| Defrost-Ret-B + Defrost-Rerank v1 | 0.870 | 0.828 | 0.862 | – | – |
| Defrost-Ret-B + Defrost-Rerank v2 (default) | **0.890** | 0.832 | 0.902 | – | – |
| Defrost-Ret-B + bge-reranker-v2-m3 (568M, reference) | 0.835 | 0.843 | 0.949 | – | – |

## What did not work (so it is not in the package)

- **A fitted query router.** A depth-2 decision tree over query features was evaluated leave-one-domain-out.
  It never beat fixed modes out of domain (nDCG 0.797 vs rerank 0.856 on dev), because its splits changed per fold.
  The parameter-free `fast` rule replaced it.
- **Equal-weight fusion of all three (`all`).** It was 0.050 below rerank on dev and 0.027 above it on test.
  That instability is why it is an option, not the default.
- **Longer training with dense hard negatives (Defrost-Ret-B2).** It lost on the dev stack and on ODEX/DS-1000.
- **Section-graph expansion and query decomposition for multi-hop questions (books).** Neither gave a gain.

## Known limits

- **Private-repo domain gap.** On private product docs, Defrost-Ret-B alone scores below BM25: the gold section beats
  every other section in only 41% of questions, against 72–80% elsewhere. The reranker carries this domain.
- **Books.** Defrost-Rerank v2 still trails the larger bge-reranker-v2-m3 on books (0.902 vs 0.949; v1 was 0.862).
- **Rerank latency.** Defrost-Rerank takes about 6–9 s per query on an Apple M-series GPU (MPS) for a 40-section pool,
  against about 0.3–0.4 s for dense. `fast` skips it when BM25 and dense agree.
