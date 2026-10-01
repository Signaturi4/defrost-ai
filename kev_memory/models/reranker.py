"""Kev-Rerank: cross-encoder. Each (query, section) pair is read jointly with full bidirectional attention as
"<instruction>: <query>\n\n<section>" (query <= 64 tokens, section <= 384); a LayerNorm + linear head over the mean
pooled states gives the score. Trained listwise (1 positive vs 7 BM25 negatives)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kev_memory.models.backbone import bidirectional_mask, load_backbone, load_tokenizer
from kev_memory.models.weights import RETRIEVAL_INSTRUCTION, device as pick_device, models_dir


class _ScoreHead(torch.nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 1)

    def forward(self, ids, att):
        h = self.backbone(input_ids=ids, attention_mask=bidirectional_mask(att.bool())).last_hidden_state
        m = att.to(h.dtype)[..., None]
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.norm(v.float())).squeeze(-1)


class KevReranker:
    def __init__(self, weights: Path | None = None, device: str | None = None, max_query: int = 64, max_doc: int = 384):
        weights = weights or models_dir()
        self.device = pick_device(device)
        self.tok = load_tokenizer()
        backbone = load_backbone(weights, weights / "kev-rerank")
        self.model = _ScoreHead(backbone, backbone.config.hidden_size)
        state = torch.load(weights / "kev-rerank/head.pt", map_location="cpu")
        self.model.norm.load_state_dict(state["norm"])
        self.model.head.load_state_dict(state["head"])
        self.model.to(self.device).eval()
        self.max_query, self.max_doc = max_query, max_doc

    def _pairs(self, query: str, texts: list[str]):
        head = self.tok(f"{RETRIEVAL_INSTRUCTION}: ", add_special_tokens=False)["input_ids"]
        q = self.tok(query, add_special_tokens=False)["input_ids"][:self.max_query]
        seqs = [head + q + self.tok("\n\n" + t, add_special_tokens=False)["input_ids"][:self.max_doc] for t in texts]
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
        return ids, att

    @torch.no_grad()
    def score(self, query: str, texts: list[str], batch_size: int = 8) -> np.ndarray:
        out = []
        for b in range(0, len(texts), batch_size):
            ids, att = self._pairs(query, texts[b:b + batch_size])
            out.append(self.model(ids.to(self.device), att.to(self.device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
