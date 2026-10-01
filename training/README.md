# Training: how the models were made

This folder holds a copy of the scripts and configs that produced the shipped weights. The copies are exact, and
`source/` keeps the original package layout (`kev_graph/...`), so every script runs unchanged from `training/source/`.
The retrieval library in `kev_memory/` does not import anything from here.

All four models start from the same backbone, Qwen2.5-0.5B (revision `060db649`), made bidirectional.
The recipe follows KG-BiLM / LLM2Vec: MNTP, then CGSA, then a supervised stage.

```
Qwen2.5-0.5B ─► 1 MNTP ─► 2 CGSA ─┬─► 3 Kev-Ret-B    (dense retriever)
  (causal)       LoRA      LoRA    └─► 4 Kev-Rerank   (cross-encoder + score head)
```

| stage | output (models/) | script | config | hardware |
|---|---|---|---|---|
| 0 corpus | `corpus_mntp.txt`, `corpus_cgsa.txt` | `source/kev_graph/bilm/techdoc_corpus.py` | built in the script | laptop |
| 1 MNTP | `base-adapters/mntp` | `source/kg_bilm_experiments/run_kmp.py` via `source/modal_bilm_app.py::run_kmp` | `source/kev_graph/bilm/configs/kmp_techdoc_modal.json` | Modal GPU |
| 2 CGSA | `base-adapters/cgsa` | `source/kev_graph/bilm/kaggle/train_cgsa_ddp.py` (notebook: `kaggle_kernels/kernel-cgsa`) | `source/kev_graph/bilm/kaggle/cgsa_kaggle.json`, as run: `configs/cgsa_run_as_trained.json` | Kaggle 2×T4, 73 min |
| 3 Kev-Ret-B | `kev-ret-b` | data `source/kev_graph/bilm/sup_data.py`; trainer `source/kev_graph/bilm/kaggle/train_sup.py`; bundle `sup_bundle.py --arm B` (notebook: `kaggle_kernels/kernel-sup`) | `configs/kev_ret_dataset_manifest.json`, `configs/manifest*.json` | Kaggle 2×T4 |
| 4 Kev-Rerank | `kev-rerank` (+ `head.pt`) | `source/kev_graph/bilm/kaggle/train_rerank.py` (notebook: `kaggle_kernels/kernel-rerank`) | `configs/kev_rerank_as_trained.json` | Kaggle 2×T4 |

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

**3. Kev-Ret-B** (supervised retrieval, "arm B" = MNTP+CGSA initialisation)
- Data: 57k (query, positive, BM25 hard negative) rows from `sup_data.py`:
  - public: MS MARCO, NQ, HotpotQA, AllNLI, Quora, StackExchange
  - tech-doc: 9.7k questions synthesized from doc sections by Claude Haiku
- 13-gram leakage gate against every evaluation suite.
- Training:
  - InfoNCE, scale 20; positives and hard negatives are all-gathered across GPUs.
  - One source per step, global batch 64.
  - LoRA r16, lr 2e-4, warmup 100, linear decay, 1000 steps, fp16.
- The query instruction is attended to but excluded from the mean pool.
- Validation accuracy 0.910.

**4. Kev-Rerank** (listwise cross-encoder)
- Same backbone (MNTP+CGSA).
- Input: `"<instruction>: <query>\n\n<section>"`, query ≤ 64 tokens, section ≤ 384.
- Head: LoRA r16 + LayerNorm + linear over mean-pooled states.
- Loss: softmax cross-entropy over 1 positive + 7 BM25 negatives.
- 16 groups per step, 1500 steps, lr 1e-4 (LoRA) and 1e-3 (head).
- Validation top-1 0.934.

## Tried and not shipped

These are kept for reference.
- `mine_hard.py` + `e1_bundle.py` (Kev-Ret-B2): 3000 steps with dense hard negatives mined by Kev-Ret-B. It lost to Kev-Ret-B on the dev stack and on ODEX/DS-1000 (0.196 / 0.233 vs 0.230 / 0.257).
- Arm A (MNTP only) and arm C (public data only) of stage 3: see docs/EVALUATION.md.

## Reproducing

```sh
cd training/source
python -m kev_graph.bilm.techdoc_corpus ...            # stage 0 (reads your local doc folders; see its docstring)
modal run modal_bilm_app.py::bilm --kmp-config kmp_techdoc_modal.json --cgsa-config cgsa_techdoc_modal.json
python -m kev_graph.bilm.kaggle.build_dataset && kaggle kernels push -p ...   # stage 2 on Kaggle
python -m kev_graph.bilm.sup_data build_v2 ...          # stage 3 data
python -m kev_graph.bilm.kaggle.sup_bundle --arm B --steps 1000 && kaggle kernels push -p ../kaggle_kernels/kernel-sup
kaggle kernels push -p ../kaggle_kernels/kernel-rerank # stage 4
python ../../scripts/export_weights.py --from <runs dir>
```

`run_kmp.py` and `run_cgsa.py` come from KG-BiLM (McGill NLP, MIT; see `source/kg_bilm_experiments/LICENSE`).
