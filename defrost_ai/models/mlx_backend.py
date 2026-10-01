"""MLX backend (Apple GPU): the same merged Qwen2 backbone, run bidirectionally with MLX's fused attention.

Used automatically on Apple Silicon when `mlx` and `mlx-lm` are installed (`pip install "defrost-ai[mac]"`);
DEFROST_BACKEND=torch forces the PyTorch path. Weights come from the merged-weights cache that the PyTorch loader
writes (defrost_ai.models.backbone.merged_dir), so both backends run exactly the same parameters.

Bidirectional attention = mlx-lm's own Qwen2 transformer blocks called with a padding-only boolean mask
[B, 1, 1, L] (every real token attends to every real token) instead of the causal mask. Pooling and the score head
run in fp32 at any backbone dtype."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np


def available() -> bool:
    if os.environ.get("DEFROST_BACKEND", "").lower() == "torch":
        return False
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_lm.models.qwen2  # noqa: F401
        return True
    except ImportError:
        return False


def _dtype(name: str):
    import mlx.core as mx
    return {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[name]


class MLXBackbone:
    def __init__(self, merged: Path, dtype: str = "bf16"):
        import mlx.core as mx
        from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model
        args = ModelArgs.from_dict(json.loads((Path(merged) / "config.json").read_text()))
        self.model = Qwen2Model(args)
        self.dtype = _dtype(dtype)
        weights = mx.load(str(Path(merged) / "model.safetensors"))
        self.model.load_weights([(k, v.astype(self.dtype)) for k, v in weights.items()], strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.hidden_size = args.hidden_size

    def mean_pool(self, ids: np.ndarray, keep: np.ndarray, pool: np.ndarray | None = None):
        """ids/keep [B, L] (keep = real tokens) -> mean of the last hidden states over `pool` (default keep), fp32."""
        import mlx.core as mx
        keep_mx = mx.array(keep)
        h = self.model.embed_tokens(mx.array(ids))
        mask = keep_mx[:, None, None, :]                    # attend to every real key; padded keys blocked
        for layer in self.model.layers:
            h = layer(h, mask)
        h = self.model.norm(h).astype(mx.float32)
        m = mx.array(keep if pool is None else pool).astype(mx.float32)[..., None]
        return (h * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1.0)


class MLXScoreHead:
    """LayerNorm + Linear over the pooled states, weights from defrost-rerank/head.pt (fp32)."""

    def __init__(self, state: dict):
        import mlx.core as mx
        n, h = state["norm"], state["head"]
        self.g, self.b = mx.array(n["weight"].float().numpy()), mx.array(n["bias"].float().numpy())
        self.w, self.c = mx.array(h["weight"].float().numpy()), mx.array(h["bias"].float().numpy())
        self.eps = 1e-5                                     # torch.nn.LayerNorm default, as trained

    def __call__(self, v):
        import mlx.core as mx
        mu = v.mean(axis=-1, keepdims=True)
        var = ((v - mu) ** 2).mean(axis=-1, keepdims=True)
        x = (v - mu) / mx.sqrt(var + self.eps) * self.g + self.b
        return (x @ self.w.T + self.c)[:, 0]
