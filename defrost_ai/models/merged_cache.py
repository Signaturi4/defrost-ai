"""Merged-weights cache bookkeeping, without torch (so `defrost status` works in a torch-free install).

    ~/.cache/defrost-ai/merged/<adapter>-<key>/   config.json + model.safetensors (fp32), one per weights chain
The key hashes the base revision and every adapter file, so new weights get a new directory; gc_merged() removes
directories that no current weights use."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from defrost_ai.models.weights import BASE_REVISION, CACHE

MERGED_CACHE = CACHE / "merged"


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


def merged_name(weights: Path, own_adapter: Path) -> str:
    adapters = (weights / "base-adapters/mntp", weights / "base-adapters/cgsa", own_adapter)
    return f"{Path(own_adapter).name}-{_chain_key(adapters)}"


def current_merged_names(weights: Path) -> set[str]:
    from defrost_ai.models.weights import adapter_dir
    return {merged_name(weights, adapter_dir(weights, n)) for n in ("defrost-ret-b", "defrost-rerank")}


def gc_merged(keep: set[str], older_than_days: float = 7, log=print) -> list[str]:
    """Delete merged-weights caches (1.8 GB each) that are not for the current weights and were not used for
    `older_than_days`, plus temp dirs left by an interrupted merge (older than an hour). Returns the removed names."""
    import time
    removed, now = [], time.time()
    if not MERGED_CACHE.exists():
        return removed
    for d in MERGED_CACHE.iterdir():
        if not d.is_dir() or d.name in keep:
            continue
        age = now - d.stat().st_mtime
        if (".tmp" in d.name and age > 3600) or (".tmp" not in d.name and age > older_than_days * 86400):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    if removed:
        log(f"defrost: removed {len(removed)} unused merged-weights cache(s): {', '.join(removed)}")
    return removed


def merged_cache_size() -> tuple[int, int]:
    """(number of merged caches, total bytes)."""
    if not MERGED_CACHE.exists():
        return 0, 0
    dirs = [d for d in MERGED_CACHE.iterdir() if d.is_dir()]
    return len(dirs), sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())


def gc_current(log=print) -> list[str]:
    """gc_merged() keeping the caches of the weights that would load now (no download); no-op without weights."""
    from defrost_ai.models.weights import WeightsError, models_dir
    try:
        weights = models_dir(download=False)
    except WeightsError:
        return []
    return gc_merged(current_merged_names(weights), log=log)
