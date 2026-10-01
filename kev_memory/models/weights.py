"""Where the model weights live and how they are verified.

Layout of a weights directory (see models/MANIFEST.json):

    base-adapters/mntp/      MNTP LoRA: turns causal Qwen2.5-0.5B into a bidirectional encoder
    base-adapters/cgsa/      CGSA LoRA: contrastive sentence alignment on top of MNTP
    kev-ret-b/               Kev-Ret-B: supervised retrieval LoRA (queries carry an instruction)
    kev-rerank/              Kev-Rerank: cross-encoder LoRA + head.pt (LayerNorm + linear score head)

Every model is: Qwen2.5-0.5B (pinned revision) -> merge MNTP -> merge CGSA -> merge its own LoRA.
Lookup order: $KEV_MEMORY_MODELS, <repo>/models, ~/.cache/kev-memory/models. If none has them, or the cache holds an
older release than WEIGHTS_VERSION, the release archive is downloaded into ~/.cache/kev-memory/models and checked
against a pinned sha256 (KEV_MEMORY_NO_DOWNLOAD=1 to disable)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
RETRIEVAL_INSTRUCTION = ("Given a developer question about a software project, retrieve the documentation passage that "
                         "answers it")


WEIGHTS_VERSION = "1.1.0"                                   # Kev-Rerank v2 (v1.0.0: Kev-Rerank v1)
WEIGHTS_URL = ("https://github.com/Signaturi4/defrost-ai/releases/download/"
               f"v{WEIGHTS_VERSION}/defrost-ai-weights-v{WEIGHTS_VERSION}.tar.gz")
WEIGHTS_SHA256 = "494fab8997ce01c46241dc23cc7dfa8c9c6097e38cfa8c5c40a1ef89ee7ae770"
CACHE = Path.home() / ".cache/kev-memory"


def _has_weights(d) -> bool:
    return bool(d) and (Path(d) / "MANIFEST.json").exists() and (Path(d) / "kev-ret-b/adapter_model.safetensors").exists()


def _version(d) -> str | None:
    try:
        return json.loads((Path(d) / "MANIFEST.json").read_text()).get("version")
    except (OSError, ValueError):
        return None


def models_dir(download: bool = True) -> Path:
    """$KEV_MEMORY_MODELS and <repo>/models are used as they are; the download cache must match WEIGHTS_VERSION,
    so an upgrade replaces cached weights from an older release."""
    for c in (os.environ.get("KEV_MEMORY_MODELS"), Path(__file__).resolve().parents[2] / "models"):
        if _has_weights(c):
            return Path(c)
    if _has_weights(CACHE / "models") and (_version(CACHE / "models") == WEIGHTS_VERSION or not download
                                           or os.environ.get("KEV_MEMORY_NO_DOWNLOAD")):
        return CACHE / "models"
    if download and not os.environ.get("KEV_MEMORY_NO_DOWNLOAD"):
        try:
            return download_weights()
        except Exception as e:                                      # noqa: BLE001  (offline, release missing, ...)
            if _has_weights(CACHE / "models"):
                import sys
                print(f"kev-memory: could not fetch weights v{WEIGHTS_VERSION} ({e}); using cached "
                      f"v{_version(CACHE / 'models')}", file=sys.stderr)
                return CACHE / "models"
            raise
    raise FileNotFoundError("kev-memory weights not found. Run `kev-memory download-weights`, or set KEV_MEMORY_MODELS "
                            "to a directory with MANIFEST.json.")


def download_weights(url: str = WEIGHTS_URL, sha256: str = WEIGHTS_SHA256, log=print) -> Path:
    """Fetch the release archive (~130 MB) once, verify its sha256, unpack to ~/.cache/kev-memory/models."""
    import shutil
    import tarfile
    import tempfile
    import urllib.request
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"kev-memory: downloading model weights (~130 MB, once) from {url}")
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        arc = Path(tmp) / "weights.tar.gz"
        h = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as r, open(arc, "wb") as f:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != sha256:
            raise RuntimeError(f"weights archive sha256 mismatch ({h.hexdigest()[:12]}… != {sha256[:12]}…); not installed")
        with tarfile.open(arc) as t:
            t.extractall(tmp, filter="data")
        dst = CACHE / "models"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(Path(tmp) / "models"), dst)
    res = verify(dst)
    if not res["ok"]:
        raise RuntimeError(f"downloaded weights failed verification: {res['bad']}")
    log(f"kev-memory: weights ready in {dst}")
    return dst


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
