"""The shared backbone: Qwen2.5-0.5B made bidirectional, with LoRA adapters merged in order.

Bidirectional attention is not a config switch: the model receives a 4D float mask in which every real token
attends to every other real token and padded keys are blocked (a 2D mask would silently keep causal attention)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

from defrost_ai.models.weights import BASE_MODEL, BASE_REVISION, CACHE

MERGED_CACHE = CACHE / "merged"


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


def _chain_key(adapters) -> str:
    """Identity of a merged model: base revision + the bytes of every adapter in the chain (sha256 from the weights
    MANIFEST when listed there, else hashed here)."""
    h = hashlib.sha256(BASE_REVISION.encode())
    for adapter in adapters:
        adapter = Path(adapter)
        manifest = next((p / "MANIFEST.json" for p in adapter.parents if (p / "MANIFEST.json").exists()), None)
        files = json.loads(manifest.read_text())["files"] if manifest else {}
        for f in sorted(adapter.glob("adapter_*")):
            rel = str(f.relative_to(manifest.parent)) if manifest else ""
            h.update(f.name.encode())
            h.update((files[rel]["sha256"] if rel in files else hashlib.sha256(f.read_bytes()).hexdigest()).encode())
    return h.hexdigest()[:16]


def merged_dir(weights: Path, own_adapter: Path) -> Path:
    """Directory of the merged model (config.json + model.safetensors, fp32): base -> MNTP -> CGSA -> own adapter.
    Built once with peft and kept in ~/.cache/defrost-ai/merged/<name>-<key>; the MLX backend reads it too."""
    adapters = (weights / "base-adapters/mntp", weights / "base-adapters/cgsa", own_adapter)
    cache = MERGED_CACHE / f"{Path(own_adapter).name}-{_chain_key(adapters)}"
    if not (cache / "config.json").exists():
        _merge(adapters).save_pretrained(tmp := cache.with_name(cache.name + f".tmp{os.getpid()}"),
                                         safe_serialization=True)
        if cache.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            os.replace(tmp, cache)
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
