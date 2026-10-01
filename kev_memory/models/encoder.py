"""Kev-Ret-B: dense retriever. Documents are embedded as "<heading path>\n<text>"; queries are prefixed with the
retrieval instruction, which is attended to but excluded from the mean pool. Vectors are L2-normalised."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kev_memory.models.backbone import bidirectional_mask, load_backbone, load_tokenizer
from kev_memory.models.weights import RETRIEVAL_INSTRUCTION, device as pick_device, models_dir

ENCODER_ID = "kev-ret-b"


class KevRetriever:
    def __init__(self, weights: Path | None = None, device: str | None = None, max_tokens: int = 512):
        weights = weights or models_dir()
        self.device = pick_device(device)
        self.tok = load_tokenizer()
        self.model = load_backbone(weights, weights / "kev-ret-b").to(self.device).eval()
        self.max_tokens = max_tokens

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._encode(texts, batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query], 16, prefix=f"{RETRIEVAL_INSTRUCTION}: ")[0]

    @torch.no_grad()
    def _encode(self, texts: list[str], batch_size: int, prefix: str | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        n_prefix = len(self.tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))     # length-sorted batches: little padding
        vecs = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for b in range(0, len(texts), batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok([(prefix or "") + texts[i] for i in idx], padding=True, truncation=True,
                           max_length=self.max_tokens + n_prefix, return_tensors="pt", padding_side="right").to(self.device)
            keep = enc["attention_mask"].bool()
            h = self.model(input_ids=enc["input_ids"], attention_mask=bidirectional_mask(keep)).last_hidden_state
            pool = keep.clone()
            pool[:, :n_prefix] = False
            v = (h * pool[..., None]).sum(1) / pool.sum(1, keepdim=True).clamp(min=1)
            vecs[idx] = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
        return vecs
