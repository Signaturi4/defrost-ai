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
| held-out (106) | 0.703 | 0.803 | 0.799 | 0.800 | 0.814 |
| private repos (69) | 0.684 | 0.667 | 0.740 | 0.804 | 0.814 |

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

## Dev and other suites (used for choices)

| system | held-out dev | private dev | books | ODEX | DS-1000 |
|---|---|---|---|---|---|
| BM25 | 0.711 | 0.678 | 0.804 | 0.134 | 0.135 |
| bge-small-en-v1.5 | – | 0.515 | 0.829 | 0.196 | 0.287 |
| bge-base-en-v1.5 | 0.799 | – | – | 0.222 | 0.273 |
| Kev-Ret-B dense | **0.886** | 0.541 | 0.855 | **0.230** | 0.257 |
| Kev-Ret-B + Kev-Rerank | 0.870 | 0.828 | 0.862 | – | – |
| Kev-Ret-B + bge-reranker-v2-m3 (568M, reference) | 0.835 | 0.843 | 0.949 | – | – |

## What did not work (so it is not in the package)

- **A fitted query router.** A depth-2 decision tree over query features was evaluated leave-one-domain-out.
  It never beat fixed modes out of domain (nDCG 0.797 vs rerank 0.856 on dev), because its splits changed per fold.
  The parameter-free `fast` rule replaced it.
- **Equal-weight fusion of all three (`all`).** It was 0.050 below rerank on dev and 0.027 above it on test.
  That instability is why it is an option, not the default.
- **Longer training with dense hard negatives (Kev-Ret-B2).** It lost on the dev stack and on ODEX/DS-1000.
- **Section-graph expansion and query decomposition for multi-hop questions (books).** Neither gave a gain.

## Known limits

- **Private-repo domain gap.** On private product docs, Kev-Ret-B alone scores below BM25: the gold section beats
  every other section in only 41% of questions, against 72–80% elsewhere. The reranker carries this domain.
- **Books.** Kev-Rerank trails the larger bge-reranker-v2-m3 on books (0.862 vs 0.949).
- **Rerank latency.** Kev-Rerank takes about 6–9 s per query on an Apple M-series GPU (MPS) for a 40-section pool,
  against about 0.3–0.4 s for dense. `fast` skips it when BM25 and dense agree.
