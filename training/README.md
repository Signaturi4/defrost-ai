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
- Technical prose only: local project docs, open-source docs, PEPs, 3k Stack Overflow threads and 40k CodeSearchNet docstrings.
- Documents are deduplicated and secret-scrubbed.
- Held-out benchmark repos are excluded.
- Output: MNTP 50M characters, CGSA 272k sentences.

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

## Tried and not shipped

These are kept for reference.
- `mine_hard.py` + `e1_bundle.py` (Defrost-Ret-B2): 3000 steps with dense hard negatives mined by Defrost-Ret-B. It lost to Defrost-Ret-B on the dev stack and on ODEX/DS-1000 (0.196 / 0.233 vs 0.230 / 0.257).
- Arm A (MNTP only) and arm C (public data only) of stage 3: see docs/EVALUATION.md.

## Reproducing

```sh
cd training/source
python -m defrost_graph.bilm.techdoc_corpus ...            # stage 0 (reads your local doc folders; see its docstring)
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
