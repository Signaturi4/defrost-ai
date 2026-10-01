"""kev-memory: hybrid-RAG knowledge memory (BM25 + Kev-Ret-B + Kev-Rerank, doc->code links from graphify's AST).

Imports are lazy so the stdlib-only parts (kev_memory.service.client, the Shepherd integration) work in an
environment without torch."""
__version__ = "1.0.0"
__all__ = ["build", "rollback", "Library", "register", "Memory", "Models"]


def __getattr__(name):
    if name in ("build", "rollback"):
        from kev_memory import builder as _b
        return getattr(_b, name)
    if name in ("Library", "register"):
        from kev_memory import library
        return getattr(library, name)
    if name in ("Memory", "Models"):
        from kev_memory import memory
        return getattr(memory, name)
    raise AttributeError(name)
