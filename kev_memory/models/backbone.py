"""The shared backbone: Qwen2.5-0.5B made bidirectional, with LoRA adapters merged in order.

Bidirectional attention is not a config switch: the model receives a 4D float mask in which every real token
attends to every other real token and padded keys are blocked (a 2D mask would silently keep causal attention)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kev_memory.models.weights import BASE_MODEL, BASE_REVISION


def merge_adapter(model, adapter_dir: Path):
    """Apply a LoRA adapter and merge it; refuse a partial load (every layer x target module must attach)."""
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    attached = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A"))
    targets = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())["target_modules"]
    expected = model.config.num_hidden_layers * len(targets)
    if attached != expected:
        raise RuntimeError(f"partial adapter load from {adapter_dir}: {attached} != {expected} LoRA modules")
    return peft_model.merge_and_unload()


def load_backbone(weights: Path, own_adapter: Path):
    """base (fp32, sdpa) -> MNTP -> CGSA -> own adapter, all merged."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32, attn_implementation="sdpa")
    for adapter in (weights / "base-adapters/mntp", weights / "base-adapters/cgsa", own_adapter):
        model = merge_adapter(model, adapter)
    return model


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


def bidirectional_mask(keep: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """keep: [B, L] bool (real tokens) -> [B, 1, L, L] additive mask: attend to all real keys, block padding."""
    B, L = keep.shape
    mask = torch.zeros(B, 1, L, L, device=keep.device, dtype=dtype)
    return mask.masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)
