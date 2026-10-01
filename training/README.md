# Training: how the models were made

This folder holds a copy of the scripts and configs that produced the shipped weights. The copies are exact, and
`source/` keeps the original package layout (`defrost_graph/...`), so every script runs unchanged from `training/source/`.
The retrieval library in `defrost_ai/` does not import anything from here.

All four models start from the same backbone, Qwen2.5-0.5B (revision `060db649`), made bidirectional.
The recipe follows KG-BiLM / LLM2Vec: MNTP, then CGSA, then a supervised stage.

```
Qwen2.5-0.5B ─► 1 MNTP ─► 2 CGSA ─┬─► 3 Defrost-Ret-B    (dense retriever)
  (causal)       LoRA      LoRA    └─► 4 Defrost-Rerank   (cross-encoder + score head; 4b = v2, shipped)
```

| stage | output (models/) | script | config | hardware |
|---|---|---|---|---|
| 0 corpus | `corpus_mntp.txt`, `corpus_cgsa.txt` | `source/defrost_graph/bilm/techdoc_corpus.py` | built in the script | laptop |
| 1 MNTP | `base-adapters/mntp` | `source/kg_bilm_experiments/run_kmp.py` via `source/modal_bilm_app.py::run_kmp` | `source/defrost_graph/bilm/configs/kmp_techdoc_modal.json` | Modal GPU |
| 2 CGSA | `base-adapters/cgsa` | `source/defrost_graph/bilm/kaggle/train_cgsa_ddp.py` (notebook: `kaggle_kernels/kernel-cgsa`) | `source/defrost_graph/bilm/kaggle/cgsa_kaggle.json`, as run: `configs/cgsa_run_as_trained.json` | Kaggle 2×T4, 73 min |
| 3 Defrost-Ret-B | `defrost-ret-b` | data `source/defrost_graph/bilm/sup_data.py`; trainer `source/defrost_graph/bilm/kaggle/train_sup.py`; bundle `sup_bundle.py --arm B` (notebook: `kaggle_kernels/kernel-sup`) | `configs/defrost_ret_dataset_manifest.json`, `configs/manifest*.json` | Kaggle 2×T4 |
| 4 Defrost-Rerank | `defrost-rerank` (+ `head.pt`) | `source/defrost_graph/bilm/kaggle/train_rerank.py` (notebook: `kaggle_kernels/kernel-rerank`) | `configs/defrost_rerank_as_trained.json` | Kaggle 2×T4 |

## Stage details

**0. Corpus**
- Shipped weights (up to v1.1.x): technical prose from local project docs, open-source docs, PEPs, 3k Stack Overflow
  threads and 40k CodeSearchNet docstrings. The local docs included part of the private product repos' docs.
- Documents are deduplicated and secret-scrubbed.
- Held-out benchmark repos are excluded.
- Output: MNTP 50M characters, CGSA 272k sentences.
- From the next retrain (v1.2.0), the corpus is built from allowlisted public sources only; see
  [Data provenance](#data-provenance) and `RETRAIN_PLAN.md`.

**1. MNTP** (masked next-token prediction)
- Makes the causal decoder use both directions.
- LoRA r16, 30% blank masking, lr 5e-5, WSD schedule, seq 512, batch 32.
- Held-out MNTP accuracy went from 0.192 to 0.316.

**2. CGSA** (contrastive sentence alignment, SimCSE-style InfoNCE)
- Starts from the MNTP adapter: dropout 0.3, scale 20, batch 64 gathered across GPUs, 1000 steps, lr 3e-5.
- Anisotropy went from 0.786 to 0.299.
- `train_cgsa_ddp.py` fixes a DDP bug in the reference `run_cgsa.py`: the reference forwards through `self.model`, which bypasses DDP, so the replicas diverge.

**3. Defrost-Ret-B** (supervised retrieval, "arm B" = MNTP+CGSA initialisation)
- Data: 57k (query, positive, BM25 hard negative) rows from `sup_data.py`:
  - public: MS MARCO, NQ, HotpotQA, AllNLI, Quora, StackExchange
  - tech-doc: 9.7k questions synthesized from doc sections by Claude Haiku
- 13-gram leakage gate between the generated questions and every evaluation suite (the passages and the MNTP/CGSA
  pretraining text were not gated; see docs/EVALUATION.md, Leakage).
- Training:
  - InfoNCE, scale 20; positives and hard negatives are all-gathered across GPUs.
  - One source per step, global batch 64.
  - LoRA r16, lr 2e-4, warmup 100, linear decay, 1000 steps, fp16.
- The query instruction is attended to but excluded from the mean pool.
- Validation accuracy 0.910.

**4. Defrost-Rerank** (listwise cross-encoder)
- Same backbone (MNTP+CGSA).
- Input: `"<instruction>: <query>\n\n<section>"`, query ≤ 64 tokens, section ≤ 384.
- Head: LoRA r16 + LayerNorm + linear over mean-pooled states.
- Loss: softmax cross-entropy over 1 positive + 7 BM25 negatives.
- 16 groups per step, 1500 steps, lr 1e-4 (LoRA) and 1e-3 (head).
- Validation top-1 0.934.

**4b. Defrost-Rerank v2** (shipped since weights v1.1.0; config `configs/defrost_rerank_v2_as_trained.json`)
- Warm start from Defrost-Rerank v1 (LoRA + head), then 800 steps at lr 3e-5 (head 3e-4), 50 warmup steps, Kaggle 2×T4.
- Data (`source/defrost_graph/bilm/rerank_v2_data.py`): 22.5k groups. These are v1's tech-doc questions plus
  617 changelog questions and 932 long-prose questions, with public replay (MS MARCO, NQ, HotpotQA,
  StackExchange, Quora, AllNLI). Claude Haiku wrote the new changelog and prose questions from the sections; a
  13-gram gate drops any that overlap the eval suites.
- Negatives: up to 2 same-file siblings and 2 changelog / release-note sections per question, ranked by BM25; the
  rest are v1's BM25 negatives. `rerank_v2_filter.py` drops negatives that Defrost-Ret-B scores at 0.95× the positive or
  higher, as likely false negatives.
- Validation top-1: 0.906 on the new 224-item set (v1: 0.880), 0.938 on v1's set (v1: 0.934).
- Locked test vs v1: `fast` +0.024 [+0.010, +0.041], `rerank` +0.037 [+0.019, +0.056] nDCG@10 (docs/RESULTS.md §2.2).

## Data provenance

Every training stage now reads only the sources listed in `configs/sources_allowlist.json`, and writes a manifest
next to its output. `source/defrost_graph/bilm/data_provenance.py` implements both.

**Allowlist** (`configs/sources_allowlist.json`, schema `defrost.sources/v1`):
- Each source has an `id`, `kind`, `role`, `license` and a pinned `revision`:
  - `oss`: https URL + 40-hex commit;
  - `hf`: Hugging Face `hf_id` + 40-hex revision;
  - `book`: URL + commit;
  - `url`: download URL + `sha256` of the archive.
- `role` is `train`, `heldout` (evaluation repos, never trained on) or `dev` (a dev suite, never trained on).
- `local_dir` points to a pinned clone under a base directory (`{oss}`, `{prose}`), never a home folder.
- A file outside every listed clone, or a row whose source id is not listed, stops the build (`SourceRejected`).
  The corpus builder has no local-folder roots.
- `license_verified: false` marks licenses taken from the dataset's origin rather than its card; review these before
  a commercial release.

**Private markers** (`configs/private_markers.local.json`, git-ignored, never committed):
- `path_patterns`: client names and private paths. A matching path fails the build (`PrivateDataError`), on top of
  the generic `deny_path_patterns` in the allowlist (Desktop, Documents, Downloads, `*-export`, ...).
- `content_terms`: a document that contains one, as a whole word, fails the build. The term is never printed.
- `private_corpora`: private text for the leakage gate (directories, `.jsonl`, `.txt`, memory `.sqlite` files). A
  `{"path", "match"}` entry keeps only files whose relative path matches.
- Builds are strict by default: a missing markers file is an error. `DEFROST_PROVENANCE_STRICT=0` allows a public
  rebuild without it.

**Leakage gate** (every stage):
- Measures the 13-gram overlap of each unit (a document, a sentence or a training row) with the evaluation suites
  (`eval_suites` in the allowlist) and with the private corpora.
- 13-grams found in two or more sources of the same stage are boilerplate and do not count.
- A unit with at least `private_min` (3) private 13-grams fails the stage.
- A unit with at least `eval_min` (3) eval 13-grams is dropped by the builders. `verify` mode fails on it instead.
- Reports go to `results/private/provenance/gate_<stage>.json` (git-ignored). They hold counts and unit indices,
  never text.

**Stage manifest**: `<stage file>.provenance.json`, written next to each output.

| stage | file | manifest `stage` |
|---|---|---|
| MNTP corpus | `corpus/corpus_mntp.txt` | `mntp_corpus` |
| CGSA corpus | `corpus/corpus_cgsa.txt` | `cgsa_corpus` |
| Ret-B supervised data | `sup_v1/train*.jsonl`, `val*.jsonl` | `retb_sup:train`, `retb_sup:val` (`_v2` for build_v2) |
| Rerank v1 data | `sup_v1/rerank_{train,val}.jsonl` | `rerank_v1_train:rerank_train`, `rerank_v1_val:rerank_val` |
| Rerank v2 data | `rerank_v2/rerank_{train,val,val_v1}.jsonl` | `rerank_v2:rerank_train`, ... |

```json
{
 "schema": "defrost.provenance/v1",
 "stage": "mntp_corpus",
 "file": "corpus_mntp.txt",
 "sha256": "<sha256 of the stage file>",
 "bytes": 36364013,
 "created": "2026-10-02T03:40:12",
 "allowlist_sha256": "<sha256 of the canonical allowlist JSON>",
 "private_markers": true,
 "sources": [
  {"id": "oss:django", "kind": "oss", "revision": "a013c821...", "license": "BSD-3-Clause",
   "license_verified": true, "origin": "https://github.com/django/django", "n_docs": 509, "n_chars": 6058664}
 ],
 "gate": {"ok": true, "mode": "drop_eval", "units": 55251, "private_hits": 0, "eval_hits": 23,
          "private_min": 3, "eval_min": 3, "ngram": 13, "boilerplate_grams": 2938},
 "inputs": [{"file": "docs.jsonl", "sha256": "<sha256>"}]
}
```

- `sources[]` counts only what is in the file: `n_docs` is the number of units (documents, sentences or rows)
  from that source, and `n_chars` is their size in characters.
- `gate.eval_hits` counts the units that were dropped (`drop_eval` mode). It is 0 for a file that passes `verify`.
- `inputs` records the upstream files, so a chain of stages can be checked end to end.

A release check must verify:
1. `schema` equals `defrost.provenance/v1`.
2. `sha256` matches the file.
3. `gate.ok` is true and `gate.private_hits` is 0.
4. Every `sources[].id` is in the allowlist with `role: train` and the same `revision`, and has a `license`.

`data_provenance.verify_manifest(path, allowlist)` performs these checks.

```sh
cd training/source
python -m defrost_graph.bilm.data_provenance --base oss=<OSS clones> --base prose=<book clones> check-sources
python -m defrost_graph.bilm.data_provenance verify <dir>/corpus_mntp.txt.provenance.json
python -m defrost_graph.bilm.data_provenance gate --stage mntp --file <dir>/corpus_mntp.txt     # verify-mode gate
```

## Tried and not shipped

These are kept for reference.
- `mine_hard.py` + `e1_bundle.py` (Defrost-Ret-B2): 3000 steps with dense hard negatives mined by Defrost-Ret-B. It lost to Defrost-Ret-B on the dev stack and on ODEX/DS-1000 (0.196 / 0.233 vs 0.230 / 0.257).
- Arm A (MNTP only) and arm C (public data only) of stage 3: see docs/EVALUATION.md.

## Reproducing

```sh
cd training/source
python -m defrost_graph.bilm.techdoc_corpus all            # stage 0: allowlisted sources only (TECHDOC_OSS_DIR, TECHDOC_PROSE_DIR)
modal run modal_bilm_app.py::bilm --kmp-config kmp_techdoc_modal.json --cgsa-config cgsa_techdoc_modal.json
python -m defrost_graph.bilm.kaggle.build_dataset && kaggle kernels push -p ...   # stage 2 on Kaggle
python -m defrost_graph.bilm.sup_data build_v2 ...          # stage 3 data
python -m defrost_graph.bilm.kaggle.sup_bundle --arm B --steps 1000 && kaggle kernels push -p ../kaggle_kernels/kernel-sup
kaggle kernels push -p ../kaggle_kernels/kernel-rerank # stage 4 (v1)
python -m defrost_graph.bilm.rerank_v2_data prose && python -m defrost_graph.bilm.rerank_v2_data changelog_questions
python -m defrost_graph.bilm.rerank_v2_data candidates && python -m defrost_graph.bilm.rerank_v2_filter
python -m defrost_graph.bilm.rerank_v2_data build             # stage 4b data (prose/changelog questions call `claude -p`)
kaggle kernels push -p ../kaggle_kernels/kernel-rerank-v2  # stage 4b (needs v1 as init-rerank-v1 in the dataset)
python ../../scripts/export_weights.py --from <runs dir>
```

`run_kmp.py` and `run_cgsa.py` come from KG-BiLM (McGill NLP, MIT; see `source/kg_bilm_experiments/LICENSE`).
