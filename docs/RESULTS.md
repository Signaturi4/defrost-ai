# Results: everything we measured, in one place

> [!NOTE]
> Mode names: since 1.2 the `fast` policy in this file is called `accurate` (the default), and `fast` means
> `hybrid` (no reranker).

This file pulls together every notable measurement of the defrost stack: Defrost-Ret-B retriever + Defrost-Rerank
cross-encoder + `fast` routing policy, on top of a BM25 section index and graphify's AST code graph. It is the
single place to answer: *does it work, where, by how much, and where does it fail?*
Protocol details live in [EVALUATION.md](EVALUATION.md); raw outputs are in `../results/`.

---

## 1. Bottom line

| question | answer |
|---|---|
| Does it beat the plain BM25 Text KB on locked test data? | **Yes.** nDCG@10 +0.097 (held-out repos) and +0.120 (private product repos), both 95% CIs above 0. RAGAS NVIDIA mean +0.100 [+0.054, +0.148]. |
| Is the gain from our own models? | **Yes.** The whole stack is our fine-tuned Qwen2.5-0.5B: Defrost-Ret-B + Defrost-Rerank, no third-party retriever or reranker. |
| Best single mode? | **`rerank`** is the most accurate on RAGAS (0.891 on test). |
| Which reranker ships? | **Defrost-Rerank v2** (weights v1.1.0). On the locked test it beats v1 by +0.024 [+0.010, +0.041] nDCG@10 in `fast` mode and +0.037 [+0.019, +0.056] in `rerank` mode (n = 175, paired). See §2.2. |
| Default mode? | **`fast`**. With v1 it matched rerank on nDCG@10 (0.814 vs 0.802 on test; with v2: 0.838 vs 0.839) and trails it slightly on RAGAS (−0.012, n.s.), while calling the reranker on only 46–57% of queries. It has no fitted parameters. |
| Is the package the same system that was evaluated? | **Yes.** On held-out dev the parity check passes: identical sections, identical BM25, dense and rerank rankings on 44/44 questions, identical nDCG@10 for every mode. |
| Main weaknesses | (1) the dense retriever alone loses to BM25 on private product docs; (2) reranking is slow on a laptop (6–9 s/query on MPS); (3) the reranker trails a much larger public reranker on books (0.902 vs 0.949); (4) multi-hop is unsolved; (5) ~50% of fast queries still pay rerank latency. |

---

## 2. Locked test (scored once, after every choice was frozen)

### 2.1 nDCG@10

| test set | n | BM25 | dense | hybrid | rerank | all | **fast** |
|---|---|---|---|---|---|---|---|
| held-out OSS repos (jinja, werkzeug, marshmallow) | 106 | 0.703 | 0.803 | 0.799 | 0.800 | 0.851 | **0.814** |
| private product repos (7 repos) | 69 | 0.684 | 0.667 | 0.740 | 0.804 | 0.794 | **0.814** |
| all | 175 | 0.696 | 0.749 | 0.776 | 0.802 | 0.829 | **0.814** |

Paired bootstrap, 95% CI:

| comparison | held-out | private | all |
|---|---|---|---|
| rerank (pre-registered stack) vs BM25 | **+0.097 [+0.038, +0.158]** | **+0.120 [+0.044, +0.197]** | +0.106 [+0.057, +0.154] |
| fast vs BM25 | +0.111 [+0.056, +0.167] | +0.130 [+0.063, +0.199] | +0.118 [+0.073, +0.162] |
| dense vs BM25 | +0.100 [+0.030, +0.172] | −0.017 [−0.119, +0.082] | +0.054 [−0.006, +0.114] |
| fast vs rerank | +0.014 [−0.004, +0.034] | +0.010 [−0.024, +0.043] | +0.012 [−0.005, +0.030] |

Pre-registered pass criteria (written before the test was read): all passed.

### 2.2 Defrost-Rerank v2 vs v1 (locked test, scored once on 2026-10-01)

Same candidate pools, only the reranker differs. The table in §2.1 is v1.

| test set | n | `rerank` v1 → v2 | `fast` v1 → v2 | `all` v1 → v2 |
|---|---|---|---|---|
| held-out OSS repos | 106 | 0.800 → **0.846** (+0.046 [+0.021, +0.073]) | 0.814 → **0.840** (+0.027 [+0.006, +0.050]) | 0.851 → 0.854 |
| private product repos | 69 | 0.804 → **0.829** (+0.025 [−0.001, +0.054]) | 0.814 → **0.835** (+0.021 [+0.004, +0.042]) | 0.794 → 0.789 |
| all | 175 | 0.802 → **0.839** (+0.037 [+0.019, +0.056]) | 0.814 → **0.838** (+0.024 [+0.010, +0.041]) | 0.829 → 0.828 |

Dev (`rerank`): held-out 0.870 → 0.890, private 0.828 → 0.832, books 0.862 → 0.902, long-prose dev (47 new
questions from 6 prose-heavy repos) 0.772 → 0.805. No suite got worse in a mode that uses the reranker.

Training: warm start from v1, 800 steps on Kaggle 2×T4, lr 3e-5 (head 3e-4). There are 22.5k training examples:
9.7k tech-doc, 617 changelog, 932 prose and public replay. Each tech-doc question gets up to 2 same-file siblings
and 2 changelog / release-note sections as negatives, ranked by BM25. Negatives that Defrost-Ret-B scores at 0.95× the
positive or higher are dropped as likely false negatives. In-training validation (224 items): v2 set top-1
0.880 → 0.906, v1 set 0.934 → 0.938.

### 2.2 RAGAS, NVIDIA metrics

Setup: Sonnet answers from the top-3 sections and Haiku judges. All 175 test questions × 5 modes; 744 unique
contexts were judged.

| metric | BM25 | dense | hybrid | **rerank** | all | fast |
|---|---|---|---|---|---|---|
| nv_mean | 0.791 | 0.842 | 0.855 | **0.891** | 0.875 | 0.879 |
| answer accuracy | 0.689 | 0.736 | 0.746 | **0.787** | 0.759 | 0.771 |
| context relevance | 0.891 | 0.937 | 0.941 | **0.971** | 0.960 | 0.964 |
| response groundedness | 0.794 | 0.854 | 0.877 | **0.916** | 0.907 | 0.901 |

nv_mean by suite: held-out BM25 0.775 → rerank 0.906 (+0.131 [+0.071, +0.196]); private BM25 0.816 → rerank 0.868
(+0.052 [−0.017, +0.123], **not significant**).

What this means in practice:
- An LLM answering from our context gets about **+10 points more answers right** (0.689 → 0.787).
- It stays better grounded in the sources (0.794 → 0.916).

---

## 3. Development and extra suites (used for choices; not the claim)

### 3.1 Retriever arms (dev nDCG@10, section/passage dense)

| retriever | held-out dev | private dev | ODEX | DS-1000 |
|---|---|---|---|---|
| BM25 | 0.711 | 0.678 | 0.134 | 0.135 |
| CGSA encoder only (unsupervised, KG-BiLM stages 1–2) | 0.48 | – | 0.127 | 0.146 |
| bge-small-en-v1.5 (33M) | – | 0.515 | 0.196 | **0.287** |
| bge-base-en-v1.5 (110M) | 0.799 | – | 0.222 | 0.273 |
| Defrost-Ret-A (MNTP init) | **0.899** | 0.544 | 0.216 | 0.248 |
| **Defrost-Ret-B (MNTP+CGSA init), shipped** | 0.886 | 0.531 | **0.230** | 0.257 |
| Defrost-Ret-C (public data only) | 0.806 | 0.547 | 0.171 | 0.216 |
| Defrost-Ret-B2 (3000 steps, dense negatives), rejected | 0.843–0.852 | 0.528–0.584 | 0.196 | 0.233 |

### 3.2 Rerankers over the same pool (dev nDCG@10)

| reranker | held-out dev | private dev | books |
|---|---|---|---|
| none (BM25) | 0.711 | 0.678 | 0.804 |
| Defrost-Rerank v1 (0.5B, ours) | 0.870 | 0.828 | 0.862 |
| **Defrost-Rerank v2 (0.5B, ours, default)** | **0.890** | 0.832 | 0.902 |
| bge-reranker-base (278M) | 0.717 | 0.837 | 0.907 |
| bge-reranker-v2-m3 (568M) | 0.835 | **0.843** | **0.949** |

### 3.3 Books (39 hand-written questions, 7 multi-hop; Pro Git, Eloquent JS, 500 Lines)

| system | nDCG@10 | notes |
|---|---|---|
| BM25 | 0.804 | |
| bge-small | 0.829 | |
| Defrost-Ret-B dense | 0.855 | |
| Defrost-Ret-B hybrid | 0.859 | +0.055 vs BM25, CI > 0 |
| Defrost-Ret-B + Defrost-Rerank v1 | 0.862 | |
| Defrost-Ret-B + Defrost-Rerank v2 | 0.902 | long-prose training questions; +0.040 [+0.007, +0.084] vs v1 |
| `all` (equal-weight fusion) | 0.903 | best of our modes here; best on multi-hop (0.895) |
| + bge-reranker-v2-m3 | 0.949 | reference ceiling |
| + section-graph expansion (+kg) | no gain | |
| + query decomposition | no gain | |

RAGAS on books (nv_mean): BM25 0.872, dense 0.917, hybrid 0.923, rerank 0.908, fast 0.923.

### 3.4 Routing (112 dev questions, 3 domains, leave-one-domain-out)

| policy | nDCG@10 | RAGAS nv_mean | reranker calls |
|---|---|---|---|
| always BM25 | 0.735 | 0.824 | 0% |
| always rerank | 0.856 | 0.904 | 100% |
| fitted decision tree (out-of-domain) | 0.797 | 0.853 | varies |
| **fast** (hybrid if BM25 and dense agree on top-1, else rerank) | 0.854 | **0.911** | 46% |

---

## 4. Diagnostics (why it behaves as it does)

- **Ranking is the bottleneck, not recall.**
  - The 40-section candidate pool contains a gold section for 97–100% of questions.
  - Per-question best-mode ceilings are 0.95 / 0.86 / 0.96 (held-out / private / books), against 0.87 / 0.83 / 0.86 achieved.
- **Where the reranker errs.**
  - It promotes CHANGELOG / "upgrading" sections (3 of the 9 worst held-out questions).
  - It promotes sibling sections with generic headings ("The Pros" of the wrong protocol, "Exercises").
  - It misses very short gold sections.
- **Private domain gap.** On private docs the gold section beats every other section by cosine in only 41% of
  questions, against 72–80% elsewhere. Dense alone scores 0.16–0.18 on 16–60-line gold sections there.
- **Calibration.**
  - Dense top-1 confidence after temperature scaling (T ≈ 0.6) has ECE 0.03–0.07 on held-out and books, AUROC ≈ 0.8.
  - On its 50% most confident questions, top-1 is correct 91–95% of the time. That is usable for abstention or adaptive k.
  - Defrost-Rerank scores are not calibrated on private docs (ECE ≈ 0.2).
- **Embedding geometry.** Mean cosine between random sections is 0.10–0.19, so there is no collapse (CGSA reduced anisotropy from 0.786 to 0.299).
- **Answer support.** The top-3 context contains at least half of the reference answer's content words for 86–97% of questions.
- **Label noise.** A few "misses" are defensible answers found in a different section than the single gold span
  (heldout-062, heldout-076, js-01, and two private questions). Single-span gold slightly understates every system.

---

## 5. Where it is good

- **Paraphrased how/why questions over technical docs (dev).**
  - Held-out prose questions: BM25 0.621 → dense 0.855 → rerank 0.820.
  - Dense alone is the best mode on held-out repos (0.885 dev, 0.803 test).
- **Identifier-heavy questions when reranked (dev).** Held-out identifier questions score 0.949 with rerank.
- **Grounded answers.** Context relevance 0.97 and groundedness 0.92 on test; fewer "not in the docs" failures.
- **Code-aware results.** Each section carries the code nodes it names (1,680 exact doc→code links on the held-out repos).
- **Small and local.** 0.5B backbone, 141 MB of adapters, runs on a laptop.
- **Incremental updates.** An unchanged rebuild re-encodes 0 sections; a first build of 900 sections plus a 6.5k-node code graph takes 174 s on MPS.
- **Public code-doc retrieval.** Defrost-Ret-B beats bge-base on ODEX (0.230 vs 0.222). It trails bge-small on DS-1000 (0.257 vs 0.287).

## 6. Where it is bad

- **Dense-only on unfamiliar private docs.** Short, identifier-dense internal notes: dense 0.54 vs BM25 0.68 on dev,
  0.667 vs 0.684 on test. Do not use `mode=dense` there; `fast` / `rerank` fix it (0.81).
- **Latency.** Defrost-Rerank takes about 6.5–9.5 s/query on Apple MPS for 40 candidates. BM25 takes 10–30 ms and dense 0.3–0.4 s.
  `fast` avoids the reranker on about half of queries, but its tail latency is still the reranker's.
- **Versus bigger rerankers on books.** bge-reranker-v2-m3 (568M) scores 0.949 vs our 0.902 (v1: 0.862). v2 added 932
  long-prose questions, which closed about half of the gap.
- **Multi-hop.** Graph expansion and query decomposition did not help. `all` fusion did best (0.895, n = 7), which is too few questions to trust.
- **RAGAS gain on private repos is not significant** (+0.052, CI crosses 0), even though nDCG is (+0.120).
- **Mode instability.** `all` swung from −0.050 (dev) to +0.027 (test) vs rerank, so it is not the default.

## 7. What did not work (tried, measured, dropped)

| attempt | result |
|---|---|
| Unsupervised KG-BiLM encoder (MNTP+CGSA) as a retriever | 0.48 nDCG on held-out dev, far below BM25 0.711; the supervised stage was necessary |
| Label-conditioned span extractor for a doc knowledge graph (P3) | typed F1 0.33–0.38, failed the R1 gate |
| Longer training with self-mined dense hard negatives (Defrost-Ret-B2) | lost on dev stack and on ODEX/DS-1000 |
| Fitted query router (decision tree) | overfit; below fixed modes out of domain |
| Section-graph neighbour expansion for multi-hop (books) | no gain |
| Query decomposition for compound questions | no gain |
| Doc-trust weighting / passage layer / inferred attachments | not part of the evaluated winner; left out of the package |

## 8. Reliability of the numbers

- **Freezing and splits.** Suites are frozen with sha256. Dev and test are split by hash, and the test was read once.
- **Leakage.** A 13-gram gate ran between the generated training questions and every suite. The pretraining text
  (MNTP, CGSA) and some supervised rows include part of the private product repos' docs, so the private-repo
  results are in-domain. The held-out OSS repos, the books and the e2e repos were not in any training corpus we built.
- **Statistics.** Paired bootstrap, 5000 resamples; confidence intervals are reported, not just means.
- **Judge failures.** The first RAGAS test pass was **invalid**: 617 silent judge failures, scored (0, 0.5, 0.5).
  They were detected by score-tuple counts per batch and re-judged after the judge wrapper was hardened.
  Batch means are now uniform (accuracy 0.63–0.78 per batch).
- **Question authorship.**
  - Held-out and private questions were LLM-generated from sections (LLM-verified), which favours lexical overlap and so BM25.
  - Books questions are hand-written, with verbatim evidence quotes.
- **Sample sizes.** Test n = 175 (106 + 69); books n = 39; multi-hop n = 7. Small slices carry wide CIs.
- **Parity.** The shipped package reproduces the research numbers exactly on held-out dev
  (`scripts/check_parity.py`: PARITY OK).

## 9. Recommendations

1. **Mode choice.**
   - Default to `fast`.
   - Use `rerank` when answer quality matters more than latency.
   - Use `bm25` for exact strings or error messages.
   - Avoid `dense`-only on private or internal notes.
2. **Model work.** Done on 2026-10-01:
   - Defrost-Rerank v2 with same-file sibling and changelog hard negatives (§2.2; now the default).
   - Longer-prose training data for books-style corpora (part of v2; books 0.862 → 0.902).
   - Adaptive k from the calibrated dense confidence (`k="auto"`).
   - A 1.5B backbone: tested, not a drop-in. The adapters do not fit Qwen2.5-1.5B (hidden 1536, 28 layers), and the raw
     1.5B model scores 0.027 nDCG@10 on held-out dev against 0.885 for Defrost-Ret-B, so it means retraining all four
     stages ([DESIGN_NOTES.md](DESIGN_NOTES.md)).

   Next, in order of expected value: RAGAS re-run with v2, a FAQ-style eval with unanswerable questions
   ([FAQ_CHATBOT.md](FAQ_CHATBOT.md)), reranker latency.
3. **Latency.** Batch or quantize the reranker, or run it on CUDA. The MPS figures above are the worst case.
