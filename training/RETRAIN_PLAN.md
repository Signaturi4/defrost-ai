# Retrain plan: v1.2.0 from allowlisted public data

Status (2026-10-02): **prep only.** The data pipeline and the gates are implemented and dry-run locally on CPU.
No training job has been launched, no dataset uploaded and no Claude-token job run.

## Why

The MNTP / CGSA corpus behind the shipped weights (up to v1.1.x) came from a builder that walked local home folders.
Part of the private product repos' docs entered pretraining that way. The supervised and reranker data was built
from the same cleaned documents, so it was affected too. A name-based exclusion missed documents whose folder
names did not match. v1.2.0 retrains every stage from allowlisted public sources only. See
[README § Data provenance](README.md#data-provenance) for the allowlist, the markers, the gate and the manifest
format.

## Dry run (local, CPU)

The collector, clean, corpus and gate stages ran in a separate output directory (`TECHDOC_MEGA=<...>/provenance_dryrun`).
They reused the already-downloaded public raw data (PEPs, Stack Overflow, CodeSearchNet, NER sets, SciERC) and the
pinned OSS and book clones.

| | old corpus (shipped) | clean corpus (allowlist) | change |
|---|---|---|---|
| train documents (after balancing) | 7,113 | 4,261 | −40% |
| train document characters | 50.1 M | 32.4 M | −35% |
| `corpus_mntp.txt` | 53.7 MB | 36.4 MB | −32% |
| `corpus_cgsa.txt` sentences | 272,359 | 210,115 | −23% |
| dataset sentences (CSN, NER sets, SciERC) | 50,967 | 50,967 | = |
| sources | 77 local folders + 18 OSS repos + datasets | 22 pinned clones + 6 pinned datasets, all allowlisted | |
| private-corpus overlap (≥ 3 distinct 13-grams) | **5,560 MNTP lines fail** | **0** | |
| eval overlap (≥ 3 13-grams) | 60 MNTP lines | 23 docs + 4 sentences, dropped | |

- **Mix of the clean corpus** (train characters): OSS docs 48%, datasets 33% (PEPs, Stack Overflow, CodeSearchNet),
  code-derived API references 11%, books 8%.
- **Gap fill already in the dry run.** The five allowlisted train books (Rust Book, Rust by Example, Rustonomicon,
  Hitchhiker's Guide to Python, System Design Primer) are now gathered: 511 documents, about 2.7 M characters.
- **Positive controls.** The same gate fails on every old stage:
  - old MNTP corpus: 5,560 private hits;
  - old Ret-B rows: 244;
  - old Rerank v1 groups: 862;
  - old Rerank v2 groups: 572.

  The reports are in `results/private/provenance_dryrun/` (git-ignored, counts only).
- **Before the gate.** A prototype checked raw public documents against all old local documents. It found 0
  overlaps with the private product docs. It found 7 overlaps with other local folders, which hold copies of
  public text.
- **Supervised data without new Claude calls.** `sup_data reuse` carries existing generated questions over to the
  rebuilt corpus when their section is unchanged:
  - Ret-B v1: 6,028 of 9,889 questions are kept (all 6,031 public-source questions match);
  - Ret-B v2: 10,568 of 21,358;
  - Rerank v2: 1,015 prose questions (books, all allowlisted) and 628 of 849 changelog questions.

  The dropped questions came from the private and other local sections.
- **Ret-B stage dry run** (`sup_data reuse` + `build`, CPU, no Claude calls):
  - 53,443 train + 512 val rows; 5,925 of them are tech-doc rows.
  - The old gate dropped 60 rows (49 tech-doc, 11 StackExchange). The provenance gate dropped 11 more for eval
    overlap.
  - **0 private hits.** Both manifests verify.
  - Wall time 1 h 16 min, but only 4.6 min of CPU: the build spent most of its time waiting, not computing.
  - The Rerank v1 / v2 data steps were not dry-run.

### Remaining gap and how to fill it

The clean corpus is 35% smaller. These public sources can fill the gap. Each needs a new allowlist entry with a
pinned revision, and downloading it needs network access.

| source | license | size (est.) | note |
|---|---|---|---|
| More Stack Overflow python threads (`HuggingFaceTB/stackexchange_2025_md`, already allowlisted) | CC BY-SA | +5–10 M chars | raise `SO_ROW_GROUPS` / `SO_MAX_THREADS`; capped at 12% of the corpus per project |
| CPython `Doc/` (python/cpython) | PSF-2.0 | ~15 M chars of .rst | high value: library reference prose |
| More OSS docs: numpy, pandas, scikit-learn, celery, aiohttp, mypy, poetry, sphinx | BSD / MIT / Apache | ~10–20 M chars | **exclude e2e repos** (uvicorn, cattrs, structlog) and held-out repos (jinja, werkzeug, marshmallow) |
| CodeSearchNet docstrings beyond 40k (`Nan-Do/code-search-net-python`, allowlisted) | Apache-2.0 | +2–4 M chars | raise `CSN_MAX` |

Do not add the books eval sources (Pro Git, Eloquent JavaScript, 500 Lines) or the prose dev book (Mostly Adequate
Guide). The gate would drop their overlap anyway, but they must not be listed as train sources.

## Stages, compute and gates

| # | stage | data step (local, CPU) | training | est. time | gate before the next stage (proposed) |
|---|---|---|---|---|---|
| 0 | corpus | `techdoc_corpus gather clean corpus` (allowlisted only) | — | 1 min (dry run: 40 s) | manifests verify; 0 private hits |
| 1 | MNTP | — | Modal H100, `run_kmp.py`, same config as shipped | ~0.5 h (train 10 min + image/IO) | held-out MNTP accuracy ≥ the shipped run's 0.316 − 0.01 |
| 2 | CGSA | upload the gated corpus as a **private** Kaggle dataset | Kaggle 2×T4, `train_cgsa_ddp.py`, 1000 steps | ~1.3 h | anisotropy ≤ 0.35 |
| 3 | Ret-B | `sup_data reuse` + `build` (questions reused, rows gated) | Kaggle 2×T4, `train_sup.py`, 1000 steps | ~2 h | val acc ≥ 0.89; dev nDCG within the old model's CI |
| 4 | Rerank v1 | `sup_data rerank` (gated) | Kaggle 2×T4, 1500 steps | ~2.5 h | val top-1 ≥ 0.92 |
| 4b | Rerank v2 | `rerank_v2_data candidates` → `rerank_v2_filter` → `build` (gated) | Kaggle 2×T4, 800 steps, warm start from new v1 | ~1.5 h | val top-1 ≥ v1.1.0 − 0.01 |
| 5 | release check | `scripts/release_check.py` verifies every `*.provenance.json` | — | minutes | all manifests ok; `license_verified: false` sources reviewed |

- **Total.** About 8 GPU hours: about 0.5 h on Modal and about 7.5 h of the Kaggle quota, which is 30 h a week.
  That is one quota week.
- **Rule.** A stage starts only after the previous stage's manifests verify (`data_provenance verify`) and its
  quality gate passes.

## Expected v1.2.0 release

- **Weights.** Defrost-Ret-B and Defrost-Rerank v2, retrained on allowlisted public data.
- **Manifests.** Each training file's `*.provenance.json` ships with the weights' training card.
- **Expected quality.**
  - Held-out, books and e2e suites: about the same as v1.1.x. These suites never depended on private text.
  - Private-repo suite: expect a drop. That suite was in-domain for the old weights and is now truly held out.
- **Docs.** Update `docs/EVALUATION.md` § Leakage: v1.2.0's pretraining and supervised text passed the private gate.
- **Model cards.** Record the source list and licenses, and the non-commercial terms of MS MARCO and the
  Hitchhiker's Guide.

## Needs your go-ahead

1. **Network downloads for the gap fill**: clone the extra OSS repos and CPython, and fetch more Stack Overflow row
   groups. Then add their pinned entries to the allowlist.
2. **Claude-token job (optional)**: generate questions for the ~10k new eligible sections that have none yet,
   including the books and the gap-fill repos. That is about 1,300 `claude -p` calls at 8 sections per call.
   Without it, Ret-B trains on the 6k reused questions plus the public sets.
3. **Upload** the gated corpus and the supervised data as **private** Kaggle datasets, and the corpus to the Modal
   volume.
4. **Launch** MNTP (Modal), then CGSA, Ret-B, Rerank v1 and Rerank v2 (Kaggle).
5. **License review** of the sources marked `license_verified: false`, and of the non-commercial ones (MS MARCO,
   Hitchhiker's Guide), before a commercial release.
6. **Publish** v1.2.0 weights after the release check.
