"""Where the model weights live and how they are verified.

Layout of a weights directory (see models/MANIFEST.json):

    base-adapters/mntp/      MNTP LoRA: turns causal Qwen2.5-0.5B into a bidirectional encoder
    base-adapters/cgsa/      CGSA LoRA: contrastive sentence alignment on top of MNTP
    defrost-ret-b/               Defrost-Ret-B: supervised retrieval LoRA (queries carry an instruction)
    defrost-rerank/              Defrost-Rerank: cross-encoder LoRA + head.pt (LayerNorm + linear score head)

Every model is: Qwen2.5-0.5B (pinned revision) -> merge MNTP -> merge CGSA -> merge its own LoRA.
Lookup order: $DEFROST_MODELS, <repo>/models, ~/.cache/defrost-ai/models. If none has them, or the cache holds an
older release than WEIGHTS_VERSION, the release archive is downloaded into ~/.cache/defrost-ai/models and checked
against a pinned sha256 (DEFROST_NO_DOWNLOAD=1 to disable). If that fails, loading fails loudly (WeightsError);
older cached weights run only with models.allow_older_weights = true, and then every search result says so."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
ENCODER_ID = "defrost-ret-b"   # stored with the section vectors: keep stable
RETRIEVAL_INSTRUCTION = ("Given a developer question about a software project, retrieve the documentation passage that "
                         "answers it")


WEIGHTS_VERSION = "1.1.0"                                   # Defrost-Rerank v2 (v1.0.0: Defrost-Rerank v1)
WEIGHTS_URL = ("https://github.com/Signaturi4/defrost-ai/releases/download/"
               f"v{WEIGHTS_VERSION}/defrost-ai-weights-v{WEIGHTS_VERSION}.tar.gz")
WEIGHTS_SHA256 = "494fab8997ce01c46241dc23cc7dfa8c9c6097e38cfa8c5c40a1ef89ee7ae770"
CACHE = Path.home() / ".cache/defrost-ai"
if not CACHE.exists() and (Path.home() / ".cache/kev-memory").exists():    # cache from before the rename
    CACHE = Path.home() / ".cache/kev-memory"
LEGACY_NAMES = {"defrost-ret-b": "kev-ret-b", "defrost-rerank": "kev-rerank"}    # layout of the v1.1.0 archive


def adapter_dir(weights, name: str) -> Path:
    """weights/<name>, or its pre-rename folder name when only that one exists."""
    new, old = Path(weights) / name, Path(weights) / LEGACY_NAMES.get(name, name)
    return old if not new.exists() and old.exists() else new


def _has_weights(d) -> bool:
    return bool(d) and (Path(d) / "MANIFEST.json").exists() and (adapter_dir(d, "defrost-ret-b") / "adapter_model.safetensors").exists()


def _version(d) -> str | None:
    try:
        return json.loads((Path(d) / "MANIFEST.json").read_text()).get("version")
    except (OSError, ValueError):
        return None


class WeightsError(RuntimeError):
    """The pinned weights are not available. Raised instead of silently running older weights."""


WARNING: str | None = None          # set when older weights are used on purpose (models.allow_older_weights)


def _allow_older() -> bool:
    try:
        from defrost_ai import settings
        return bool(settings.get("models.allow_older_weights"))
    except Exception:                                               # noqa: BLE001
        return os.environ.get("DEFROST_ALLOW_OLDER_WEIGHTS", "").lower() in ("1", "true", "yes")


def models_dir(download: bool = True) -> Path:
    """The weights directory to load. $DEFROST_MODELS and <repo>/models are used as they are (an explicit choice);
    the download cache must hold WEIGHTS_VERSION. A missing or older cache is downloaded; if that fails, this
    raises WeightsError naming the version and the fix, unless models.allow_older_weights is on, in which case the
    older cache is used and every search result carries a warning line (WARNING).
    download=False: no network; returns the cache whatever its version (for status / verification only)."""
    global WARNING
    for c in (os.environ.get("DEFROST_MODELS"), Path(__file__).resolve().parents[2] / "models"):
        if _has_weights(c):
            return Path(c)
    cached = _has_weights(CACHE / "models")
    if cached and (not download or _version(CACHE / "models") == WEIGHTS_VERSION):
        return CACHE / "models"
    if not download:
        raise WeightsError(_missing_message("not downloaded"))
    err = "downloads disabled (DEFROST_NO_DOWNLOAD)"
    if not os.environ.get("DEFROST_NO_DOWNLOAD"):
        try:
            return download_weights()
        except Exception as e:                                      # noqa: BLE001  (offline, release missing, ...)
            err = f"download failed: {e}"
    if cached and _allow_older():
        WARNING = (f"weights v{_version(CACHE / 'models')} in use, expected v{WEIGHTS_VERSION} "
                   f"(models.allow_older_weights is on; {err})")
        import sys
        print(f"defrost: WARNING {WARNING}", file=sys.stderr)
        return CACHE / "models"
    have = f"cache has v{_version(CACHE / 'models')}" if cached else "nothing cached"
    raise WeightsError(_missing_message(f"{err}; {have}"))


def _missing_message(why: str) -> str:
    return (f"defrost needs model weights v{WEIGHTS_VERSION} ({why}).\n"
            f"  expected: {WEIGHTS_URL}\n"
            f"  fix: run `defrost download-weights`, or set DEFROST_MODELS to a directory with MANIFEST.json "
            f"(v{WEIGHTS_VERSION}).\n"
            f"  to run older cached weights anyway: `defrost config models.allow_older_weights true` "
            f"(every result will say so).")


def status() -> dict:
    """What `defrost status` shows: which weights would load, their version, and whether it is the pinned one."""
    try:
        d = models_dir(download=False)
    except WeightsError:
        return {"dir": None, "version": None, "expected": WEIGHTS_VERSION, "matches": False}
    v = _version(d)
    src = ("DEFROST_MODELS" if os.environ.get("DEFROST_MODELS") and Path(os.environ["DEFROST_MODELS"]) == d
           else "cache" if d == CACHE / "models" else "repo")
    return {"dir": str(d), "version": v, "expected": WEIGHTS_VERSION, "matches": v == WEIGHTS_VERSION, "source": src}


def download_weights(url: str | None = None, sha256: str | None = None, log=print) -> Path:
    """Fetch the release archive (~130 MB) once, verify its sha256, unpack to ~/.cache/defrost-ai/models.
    url/sha256 default to WEIGHTS_URL/WEIGHTS_SHA256, read at call time (not bound when the module loads)."""
    url, sha256 = url or WEIGHTS_URL, sha256 or WEIGHTS_SHA256
    import shutil
    import tarfile
    import tempfile
    import urllib.request
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"defrost: downloading model weights (~130 MB, once) from {url}")
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
    log(f"defrost: weights ready in {dst}")
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
