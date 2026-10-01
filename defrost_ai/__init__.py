"""defrost: hybrid-RAG knowledge memory (BM25 + Defrost-Ret-B + Defrost-Rerank, doc->code links from graphify's AST).

Imports are lazy so the stdlib-only parts (defrost_ai.service.client, the Shepherd integration) work in an
environment without torch."""
import os as _os

__version__ = "1.1.0"

# Before 1.2 the package was `kev-memory`: honour its environment variables and data folder for one minor version.
for _k, _v in list(_os.environ.items()):
    if _k.startswith(("KEV_MEMORY_", "KEV_")):
        _os.environ.setdefault("DEFROST_" + _k.split("_", 2 if _k.startswith("KEV_MEMORY_") else 1)[-1], _v)
if "DEFROST_HOME" not in _os.environ and not _os.path.isdir(_os.path.expanduser("~/.defrost-ai")) \
        and _os.path.isdir(_os.path.expanduser("~/.kev-memory")):
    _os.environ["DEFROST_HOME"] = _os.path.expanduser("~/.kev-memory")


def build_id() -> str:
    """Version + hash of the package sources: a running service with a different build id is restarted by clients."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha1(__version__.encode())
    for f in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(f.read_bytes())
    return f"{__version__}+{h.hexdigest()[:10]}"
__all__ = ["build", "rollback", "Library", "register", "Memory", "Models"]


def __getattr__(name):
    if name in ("build", "rollback"):
        from defrost_ai import builder as _b
        return getattr(_b, name)
    if name in ("Library", "register"):
        from defrost_ai import library
        return getattr(library, name)
    if name in ("Memory", "Models"):
        from defrost_ai import memory
        return getattr(memory, name)
    raise AttributeError(name)
