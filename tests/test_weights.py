"""Weights fail loudly (no silent fallback to an older cache), the opt-in warning, merged-cache GC, dtype per platform.
No weights, no network, no torch needed (the dtype test skips without torch)."""
import json
import os
import time

import pytest


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An isolated download cache holding v1.0.0 weights; downloads fail; no DEFROST_MODELS, no <repo>/models."""
    from defrost_ai.models import merged_cache, weights
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path / "home"))
    for k in ("DEFROST_MODELS", "DEFROST_NO_DOWNLOAD", "DEFROST_ALLOW_OLDER_WEIGHTS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(weights, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(weights, "WARNING", None)
    monkeypatch.setattr(merged_cache, "MERGED_CACHE", tmp_path / "cache/merged")
    real = weights._has_weights
    monkeypatch.setattr(weights, "_has_weights", lambda d: bool(d) and str(tmp_path) in str(d) and real(d))

    def offline(*a, **kw):
        raise OSError("offline")
    monkeypatch.setattr(weights, "download_weights", offline)
    models = tmp_path / "cache/models"
    (models / "defrost-ret-b").mkdir(parents=True)
    (models / "defrost-ret-b/adapter_model.safetensors").write_bytes(b"x")
    (models / "MANIFEST.json").write_text(json.dumps({"version": "1.0.0", "files": {}}))
    return models


def test_older_cache_and_failed_download_raise_a_clear_error(cache):
    from defrost_ai.models import weights
    with pytest.raises(weights.WeightsError) as e:
        weights.models_dir()
    msg = str(e.value)
    assert weights.WEIGHTS_VERSION in msg and weights.WEIGHTS_URL in msg
    assert "defrost download-weights" in msg and "DEFROST_MODELS" in msg and "v1.0.0" in msg
    assert weights.WARNING is None


def test_status_reports_the_mismatch(cache):
    from defrost_ai.models import weights
    s = weights.status()
    assert s["version"] == "1.0.0" and s["expected"] == weights.WEIGHTS_VERSION and not s["matches"]


def test_allow_older_weights_runs_them_with_a_warning_on_every_answer(cache, monkeypatch):
    from defrost_ai import settings
    from defrost_ai.memory import Memory
    from defrost_ai.models import weights
    settings.set_("models.allow_older_weights", "true")
    assert settings.get("models.allow_older_weights") is True
    assert weights.models_dir() == cache
    assert "v1.0.0" in weights.WARNING and weights.WEIGHTS_VERSION in weights.WARNING
    text = Memory.context({"hits": [], "weights_warning": weights.WARNING})
    assert text.startswith("! weights v1.0.0 in use")


def test_explicit_models_dir_is_used_as_is(cache, monkeypatch):
    from defrost_ai.models import weights
    monkeypatch.setenv("DEFROST_MODELS", str(cache))                 # an explicit choice: no version check
    assert weights.models_dir() == cache and weights.WARNING is None


def test_bool_setting_coercion(tmp_path, monkeypatch):
    from defrost_ai import settings
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path))
    monkeypatch.delenv("DEFROST_ALLOW_OLDER_WEIGHTS", raising=False)
    assert settings.get("models.allow_older_weights") is False
    monkeypatch.setenv("DEFROST_ALLOW_OLDER_WEIGHTS", "0")
    assert settings.get("models.allow_older_weights") is False
    monkeypatch.setenv("DEFROST_ALLOW_OLDER_WEIGHTS", "yes")
    assert settings.get("models.allow_older_weights") is True


def test_gc_keeps_current_and_recent_merged_caches(tmp_path, monkeypatch):
    from defrost_ai.models import merged_cache
    root = tmp_path / "merged"
    monkeypatch.setattr(merged_cache, "MERGED_CACHE", root)
    old = time.time() - 30 * 86400
    for name, age in (("cur-a", old), ("stale-b", old), ("recent-c", None), ("x.tmp1", time.time() - 7200),
                      ("y.tmp2", None)):
        (root / name).mkdir(parents=True)
        (root / name / "model.safetensors").write_bytes(b"0" * 10)
        if age:
            os.utime(root / name, (age, age))
    removed = merged_cache.gc_merged({"cur-a"}, log=lambda m: None)
    assert sorted(removed) == ["stale-b", "x.tmp1"]
    assert sorted(p.name for p in root.iterdir()) == ["cur-a", "recent-c", "y.tmp2"]
    assert merged_cache.merged_cache_size() == (3, 30)


def test_merged_cache_name_changes_with_the_weights(tmp_path):
    from defrost_ai.models import merged_cache
    w = tmp_path / "models"
    for d in ("base-adapters/mntp", "base-adapters/cgsa", "defrost-rerank"):
        (w / d).mkdir(parents=True)
        (w / d / "adapter_model.safetensors").write_bytes(b"a")
    a = merged_cache.merged_name(w, w / "defrost-rerank")
    (w / "defrost-rerank/adapter_model.safetensors").write_bytes(b"b")
    assert a != merged_cache.merged_name(w, w / "defrost-rerank") and a.startswith("defrost-rerank-")


def test_torch_dtype_per_platform(monkeypatch):
    torch = pytest.importorskip("torch")
    from defrost_ai.models.reranker import _dtype
    monkeypatch.delenv("DEFROST_RERANK_DTYPE", raising=False)
    assert _dtype(torch.device("cuda")) == torch.bfloat16
    assert _dtype(torch.device("cpu")) == torch.float32
    assert _dtype(torch.device("mps")) == torch.float32                 # torch on a Mac: fp32 debug path
    monkeypatch.setenv("DEFROST_RERANK_DTYPE", "bf16")
    assert _dtype(torch.device("mps")) == torch.bfloat16
