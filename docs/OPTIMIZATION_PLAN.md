# Latency plan: under 3 s per search, same quality

Status: plan only (2026-10-02). Nothing below is implemented or quality-tested yet.

## 1. Where the time goes today

Measured with `scripts/profile_latency.py` (Apple M5, macOS 27.2, torch 2.14 MPS, fp32, Defrost-Rerank v2;
parity-heldout memory with 900 sections; 3 dev questions, `mode=rerank`). Raw: `results/private/latency/baseline_n3.json`.

| stage (warm, per query) | median | share |
|---|---|---|
| BM25 (SQLite FTS5) | 1.6 ms | 0% |
| query embedding (Defrost-Ret-B, 45 tokens) | 95 ms | 1.3% |
| dense dot product + top-k | 0.2 ms | 0% |
| SQLite section fetches | 0.8 ms | 0% |
| reranker tokenize + copy | 42 ms | 0.6% |
| **reranker forward** (37 pairs, 5 batches of 8) | **7,089 ms** | **98.8%** |
| **total** | **7,177 ms** | |

| cold start (once per service start) | time |
|---|---|
| import torch + transformers | 2.4 s |
| load retriever (download check, base, merge 3 LoRAs, to MPS) | 8.5 s |
| load reranker (same) | 5.0 s |
| first query (MPS kernel warm-up) | 8.1 s |

What the numbers mean:
- **The reranker forward pass is the whole problem.** Everything else together is about 140 ms.
- **The reranker wastes work on padding.** A query sends 27–37 pairs of 303 tokens on average (446 max), so 8–12k real
  tokens. Padding inside each batch of 8 raises that to 12–16k, about 40% extra.
- **The GPU runs far below capacity.** The reranker reaches about 1.1 TFLOP/s on real tokens (2 × 0.36B non-embedding
  parameters × tokens / time; our arithmetic). That is low for an M5 GPU and typical of fp32 PyTorch on MPS.
- **To reach 3 s,** the forward pass needs about 2.5× more speed. To reach 1 s, it needs about 7×.

## 2. Core concepts

1. **Measure before and after every step, with one fixed protocol.** Use 20+ warm queries and report the median and
   p90. Add a parity check against the fp32 path and the nDCG gate (§5).
2. **Do less work before doing work faster.** This means removing padding, avoiding unneeded passes (reuse cached
   scores) and skipping the reranker when it is not needed (`fast` already does this for about half of queries).
3. **Use the hardware's native precision.** Use bf16 on the Apple GPU. Avoid fp16: Qwen overflows in fp16, and its large
   negative mask values become -inf, which then produces NaN in the mean pool.
4. **Use the Mac-native runtime for the hot path.** MLX runs on the GPU, uses unified memory, and its fused attention
   accepts our bidirectional mask and GQA. On the M5 it also uses the GPU Neural Accelerators. Keep PyTorch as the
   Linux/CUDA path, so both backends compute the same function and are checked against each other.
5. **Pay fixed costs once.** Merge the LoRA adapters once at install time instead of at every start. Warm the kernels
   when the service starts, not on the user's first query.
6. **Make lossy options opt-in.** Smaller candidate pools, 8-bit weights, early exit and distillation each change the
   function. They ship only behind the quality gate, and only if they pass it.

## 3. What applies, and what does not

| option | expected effect on the reranker | quality risk | effort | verdict |
|---|---|---|---|---|
| Merged checkpoint on disk (safetensors), loaded directly | cold start −10 s or more | none (same weights) | 1 h | **do** |
| Warm-up pass at service start | first query 8 s → warm speed | none | 15 min | **do** |
| Length-sorted buckets + 1–2 large batches instead of 5×8 | about −30% (removes most of the 40% padding) | none (same math) | 1 h | **do** |
| bf16 on MPS with NaN-safe pooling | about 2× (estimate) | low; needs parity check | 2 h | **do** |
| **MLX port** (Qwen2 forward with padding mask, mean pool, LayerNorm + Linear), bf16 | about 5–10× vs today (estimate, from Apple's M5 prefill and MLX embedding benchmarks) | low; needs parity check | 4–8 h | **do; main lever** |
| Score cache keyed by (query, section hash) | repeated queries free | none | 1 h | **do** |
| Concurrency: BM25 + query embedding in parallel; tokenize the next bucket while the GPU runs | −10–50 ms | none | 1 h | do last; small |
| Smaller pool (K = 20–30 instead of 40) | up to −40% | **yes**; gate | 1 h + eval | candidate |
| 8-bit MLX weights | +20–40% on top of bf16 (estimate) | **yes**; gate (4-bit is known to break Qwen embedders) | 1 h + eval | candidate |
| Truncate sections at 256 tokens instead of 384 | about −15% (estimate) | **yes**; gate | eval only | candidate |
| Early exit / fewer layers, distilled student | 2–3.5× | **yes**; needs retraining | days | later, if still needed |
| Core ML / Neural Engine | none: 140 ms per 512-token item, about today's speed | — | days | skip |
| llama.cpp GGUF / ONNX Runtime with CoreML | unknown; our score head is unsupported, and Metal rerank bugs are open | high | 1–2 days | skip |
| torch.compile on MPS | unproven (prototype) | — | — | skip |
| Faster BM25 or vector search (FAISS, usearch) | none: these stages take under 2 ms | — | — | skip |

## 4. Step-by-step plan

Each step: implement → profile (20 queries) → parity → nDCG gate → keep or revert. The steps are cumulative.

| step | change | latency target (rerank, warm) | gate |
|---|---|---|---|
| 0 | Baseline at n = 20 on heldout + books + general_crm; record dev nDCG for fp32 | today's 7.2 s | — |
| 1 | Merged-weights cache (`~/.cache/defrost-ai/merged/<weights-version>`), warm-up at service start | warm unchanged; cold ≤ 8 s | identical scores (max abs diff ≤ 1e-5) |
| 2 | Length buckets + larger batches (fp32) | ≈ 5 s | identical rankings |
| 3 | bf16 on MPS + NaN-safe pooling | ≈ 2.5–3 s | top-1 agreement ≥ 99% vs fp32; nDCG within ±0.005 |
| 4 | MLX backend (bf16) for reranker and query encoder; auto-selected on macOS, `DEFROST_BACKEND=torch` to override | **≤ 1 s** | same as step 3, against the fp32 torch reference |
| 5 | Score cache + warm service; `fast` now pays at most the step-4 cost | repeated queries ~0 | none |
| 6 | Optional lossy knobs, each measured alone: K = 30/25/20, 8-bit, max_doc 256 | step-4 time × 0.5–0.8 | nDCG paired-bootstrap CI includes 0 and the mean drop is ≤ 0.005; otherwise off |

**Success criteria:** every mode returns in under 3 s at p90 on the M5, and cold start is under 10 s. On the dev
suites (heldout, zoop, books, prose_dev), the default path's nDCG@10 stays within ±0.005 of fp32. The locked test is
read once, at the end.

## 5. Quality gate (defined now, not run yet)

- **Reference:** fp32 PyTorch on CPU or MPS. Save the per-pair scores once per suite.
- **Parity test** (cheap, every step): run 20 queries × ~37 pairs and compare scores, top-1 agreement and Kendall τ
  of each pool ranking.
- **nDCG gate** (each step that changes numerics): `defrost benchmark` on the 4 dev suites, about 160 questions,
  with a paired bootstrap against the reference. Local only, no Claude calls.

## 6. Risks

- **MLX version drift.** Pin the `mlx` version and keep a parity test in CI, run on macOS.
- **Two backends can drift apart.** Mitigation: one shared tokenizer and pairing code, with a parity test across both
  backends.
- **Thermals.** Long batch jobs on fanless Macs throttle. Report a p90 after a 5-minute run.
- **Dependency weight.** MLX is about 30 MB and macOS-only, so make it an optional extra (`[mac]`). `install.sh` adds
  it automatically on macOS.
