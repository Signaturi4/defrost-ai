"""Where the model weights live and how they are verified.

Layout of a weights directory (see models/MANIFEST.json):

    base-adapters/mntp/      MNTP LoRA: turns causal Qwen2.5-0.5B into a bidirectional encoder
    base-adapters/cgsa/      CGSA LoRA: contrastive sentence alignment on top of MNTP
    kev-ret-b/               Kev-Ret-B: supervised retrieval LoRA (queries carry an instruction)
    kev-rerank/              Kev-Rerank: cross-encoder LoRA + head.pt (LayerNorm + linear score head)

Every model is: Qwen2.5-0.5B (pinned revision) -> merge MNTP -> merge CGSA -> merge its own LoRA.
Lookup order: $KEV_MEMORY_MODELS, <repo>/models, ~/.cache/kev-memory/models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
RETRIEVAL_INSTRUCTION = ("Given a developer question about a software project, retrieve the documentation passage that "
                         "answers it")


def models_dir() -> Path:
    candidates = [os.environ.get("KEV_MEMORY_MODELS"), Path(__file__).resolve().parents[2] / "models",
                  Path.home() / ".cache/kev-memory/models"]
    for c in candidates:
        if c and (Path(c) / "MANIFEST.json").exists():
            return Path(c)
    raise FileNotFoundError("kev-memory weights not found. Set KEV_MEMORY_MODELS to a directory with MANIFEST.json "
                            "(see README: 'Model weights').")


def verify(root: Path | None = None) -> dict:
    """Check every file against MANIFEST.json -> {"ok": bool, "bad": [paths]}"""
    root = root or models_dir()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    bad = []
    for rel, meta in manifest["files"].items():
        p = root / rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(rel)
    return {"ok": not bad, "bad": bad, "version": manifest.get("version")}


def device(name: str | None = None):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
