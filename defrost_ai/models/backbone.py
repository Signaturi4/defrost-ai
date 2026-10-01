"""The shared backbone: Qwen2.5-0.5B made bidirectional, with LoRA adapters merged in order.

Bidirectional attention is not a config switch: the model receives a 4D float mask in which every real token
attends to every other real token and padded keys are blocked (a 2D mask would silently keep causal attention)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from defrost_ai.models.merged_cache import MERGED_CACHE, _chain_key, merged_name  # noqa: F401  (torch-free part)
from defrost_ai.models.weights import BASE_MODEL, BASE_REVISION


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


def merged_dir(weights: Path, own_adapter: Path) -> Path:
    """Directory of the merged model (config.json + model.safetensors, fp32): base -> MNTP -> CGSA -> own adapter.
    Built once with peft and kept in ~/.cache/defrost-ai/merged/<name>-<key>; the MLX backend reads it too.
    Kept in fp32 on purpose: an fp16 copy is bit-identical for the MLX fp16 path but shifts the fp32 PyTorch path
    (CPU/CUDA reference, the MLX fp32 overflow retry) by up to 0.005 in score."""
    adapters = (weights / "base-adapters/mntp", weights / "base-adapters/cgsa", own_adapter)
    cache = MERGED_CACHE / merged_name(weights, own_adapter)
    if not (cache / "config.json").exists():
        _merge(adapters).save_pretrained(tmp := cache.with_name(cache.name + f".tmp{os.getpid()}"),
                                         safe_serialization=True)
        if cache.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            os.replace(tmp, cache)
    os.utime(cache)                                          # last use, for gc_merged()
    return cache


def _merge(adapters):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32, attn_implementation="sdpa")
    for adapter in adapters:
        model = merge_adapter(model, adapter)
    return model


def load_backbone(weights: Path, own_adapter: Path):
    """The merged backbone in fp32 with SDPA attention, loaded from the merged-weights cache (same weights as merging
    the adapters at start-up, without the peft cost). DEFROST_MERGED_CACHE=0 merges in memory instead."""
    from transformers import AutoModel
    if os.environ.get("DEFROST_MERGED_CACHE", "1") == "0":
        return _merge((weights / "base-adapters/mntp", weights / "base-adapters/cgsa", own_adapter))
    return AutoModel.from_pretrained(merged_dir(weights, own_adapter), dtype=torch.float32, attn_implementation="sdpa")


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


def bidirectional_mask(keep: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """keep: [B, L] bool (real tokens) -> [B, 1, L, L] additive mask: attend to all real keys, block padding."""
    B, L = keep.shape
    mask = torch.zeros(B, 1, L, L, device=keep.device, dtype=dtype)
    return mask.masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)
