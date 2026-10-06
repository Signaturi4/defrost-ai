import pytest

from defrost_ai.models import mlx_backend

mx = pytest.importorskip("mlx.core") if mlx_backend.available() else pytest.skip("MLX backend only on Apple Silicon",
                                                                                  allow_module_level=True)


def test_freed_buffers_stay_under_the_cache_limit():
    mlx_backend.limit_cache()
    for n in range(1024, 1024 + 40 * 97, 97):                   # a new shape each time, as reranking batches are
        mx.eval(mx.ones((n, 4096), dtype=mx.float32) * 2)     # 16-80 MB each, ~1.9 GB in total
    assert mx.get_cache_memory() <= mlx_backend.CACHE_MB * 2**20
