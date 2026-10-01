"""Kev-Rerank: cross-encoder. Each (query, section) pair is read jointly with full bidirectional attention as
"<instruction>: <query>\n\n<section>" (query <= 64 tokens, section <= 384); a LayerNorm + linear head over the mean
pooled states gives the score. Trained listwise (1 positive vs 7 BM25 negatives)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from kev_memory.models.backbone import bidirectional_mask, load_backbone, load_tokenizer, merged_dir
from kev_memory.models.weights import RETRIEVAL_INSTRUCTION, device as pick_device, models_dir


def _dtype(device) -> torch.dtype:
    """bf16 on Apple GPU / CUDA (2x less memory traffic; Qwen overflows in fp16, so never fp16), fp32 on CPU.
    KEV_RERANK_DTYPE=fp32|bf16 overrides."""
    name = os.environ.get("KEV_RERANK_DTYPE") or ("bf16" if device.type in ("mps", "cuda") else "fp32")
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[name]


class _ScoreHead(torch.nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 1)

    def forward(self, ids, att):
        mask = bidirectional_mask(att.bool(), dtype=self.backbone.dtype)    # padded keys blocked; no row fully masked
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = att.float()[..., None]                                          # pool + head in fp32 at any backbone dtype
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.norm(v)).squeeze(-1)


class KevReranker:
    def __init__(self, weights: Path | None = None, device: str | None = None, max_query: int = 64, max_doc: int = 384):
        from kev_memory.models import mlx_backend
        weights = weights or models_dir()
        self.device = pick_device(device)
        self.tok = load_tokenizer()
        state = torch.load(weights / "kev-rerank/head.pt", map_location="cpu")
        self.max_query, self.max_doc = max_query, max_doc
        self.backend = "mlx" if device is None and mlx_backend.available() else "torch"
        if self.backend == "mlx":                                           # Apple GPU via MLX
            self._merged = merged_dir(weights, weights / "kev-rerank")
            # fp16: same speed as bf16 on the M5 GPU and ~8x closer to fp32 (10 vs 7 mantissa bits). Qwen can overflow
            # in fp16, so a query whose scores come back non-finite is recomputed in fp32 (see score()).
            self.mlx = mlx_backend.MLXBackbone(self._merged, os.environ.get("KEV_RERANK_DTYPE", "fp16"))
            self._mlx32 = None
            self.mlx_head = mlx_backend.MLXScoreHead(state)
            self.model = None
            self.token_budget = int(os.environ.get("KEV_RERANK_TOKEN_BUDGET", 2048))   # 2048 fastest on M5 (1.6 s vs 2.0 s at 8192)
            return
        backbone = load_backbone(weights, weights / "kev-rerank")
        self.model = _ScoreHead(backbone, backbone.config.hidden_size)
        self.model.norm.load_state_dict(state["norm"])
        self.model.head.load_state_dict(state["head"])
        self.model.to(self.device).eval()
        self.dtype = _dtype(self.device)
        self.model.backbone.to(self.dtype)                                  # LayerNorm + Linear head stay fp32
        self.token_budget = int(os.environ.get("KEV_RERANK_TOKEN_BUDGET", 2048))   # padded tokens per pass (flat 512-2048 on MPS fp32; larger is slower)

    def _encode_pairs(self, query: str, texts: list[str]) -> list[list[int]]:
        """Token ids of "<instruction>: <query>\n\n<section>" for every text (query <= 64, section <= 384 tokens)."""
        head = self.tok(f"{RETRIEVAL_INSTRUCTION}: ", add_special_tokens=False)["input_ids"]
        q = self.tok(query, add_special_tokens=False)["input_ids"][:self.max_query]
        docs = self.tok(["\n\n" + t for t in texts], add_special_tokens=False)["input_ids"]
        return [head + q + d[:self.max_doc] for d in docs]

    def _pad(self, seqs: list[list[int]]):
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
        return ids, att

    def _pairs(self, query: str, texts: list[str]):
        return self._pad(self._encode_pairs(query, texts))

    def _mlx_scores(self, backbone, seqs, batches) -> np.ndarray:
        import mlx.core as mx
        outs = []
        for idx in batches:
            ids, att = self._pad([seqs[i] for i in idx])
            outs.append(self.mlx_head(backbone.mean_pool(ids.numpy(), att.numpy().astype(bool))))
        return np.array(mx.concatenate(outs), dtype=np.float32)            # one evaluation for the whole query

    @torch.no_grad()
    def score(self, query: str, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Pairs are sorted by length and cut into batches of at most `token_budget` padded tokens, so short sections
        are not padded to the longest one; scores come back in input order. One host sync per query."""
        if not texts:
            return np.zeros(0)
        seqs = self._encode_pairs(query, texts)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        batches, cur = [], []
        for i in order:
            n = len(cur) + 1
            if cur and (n * len(seqs[i]) > self.token_budget or (batch_size and n > batch_size)):
                batches.append(cur); cur = []
            cur.append(i)
        batches.append(cur)
        if self.backend == "mlx":
            flat = self._mlx_scores(self.mlx, seqs, batches)
            if not np.isfinite(flat).all():                                 # fp16 overflow: redo this query in fp32
                from kev_memory.models import mlx_backend
                self._mlx32 = self._mlx32 or mlx_backend.MLXBackbone(self._merged, "fp32")
                flat = self._mlx_scores(self._mlx32, seqs, batches)
        else:
            outs = []
            for idx in batches:
                ids, att = self._pad([seqs[i] for i in idx])
                outs.append(self.model(ids.to(self.device), att.to(self.device)))
            flat = torch.cat(outs).float().cpu().numpy()
        out = np.empty(len(texts), dtype=flat.dtype)
        out[[i for idx in batches for i in idx]] = flat
        return out
