"""Defrost-Ret-B: dense retriever. Documents are embedded as "<heading path>\n<text>"; queries are prefixed with the
retrieval instruction, which is attended to but excluded from the mean pool. Vectors are L2-normalised."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from defrost_ai.models.backbone import bidirectional_mask, load_backbone, load_tokenizer
from defrost_ai.models.weights import ENCODER_ID, RETRIEVAL_INSTRUCTION, adapter_dir, device as pick_device, models_dir



class DefrostRetriever:
    def __init__(self, weights: Path | None = None, device: str | None = None, max_tokens: int = 512):
        weights = weights or models_dir()
        self.device = pick_device(device)
        self.tok = load_tokenizer()
        self.model = load_backbone(weights, adapter_dir(weights, "defrost-ret-b")).to(self.device).eval()
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


# Hugging Face embedding models usable as the dense retriever (sentence-transformers; prefixes from each model card).
ST_ENCODERS = {
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery:", ""),
}


class STRetriever:
    """A sentence-transformers embedding model behind the DefrostRetriever interface (L2-normalised, 512 tokens)."""

    def __init__(self, name: str, device: str | None = None, max_tokens: int = 512):
        from sentence_transformers import SentenceTransformer
        hf, self.query_prefix, self.doc_prefix = ST_ENCODERS[name]
        self.model = SentenceTransformer(hf, device=pick_device(device))
        self.model.max_seq_length = max_tokens

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode([self.doc_prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0].astype(np.float32)


def encoder_name() -> str:
    from defrost_ai import settings
    return str(settings.get("retrieval.encoder"))


def make_retriever(name: str, weights: Path | None = None, device: str | None = None):
    return DefrostRetriever(weights, device) if name == ENCODER_ID else STRetriever(name, device)
