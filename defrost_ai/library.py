"""A library of domains: several memories (e.g. company code, handbook, runbooks, books) behind one search.

Registry file (default ~/.defrost-ai/domains.json, override with $DEFROST_HOME):

    {"domains": {"acme-code": {"workspace": "/path/acme.workspace.json", "description": "API + web + infra"},
                 "handbook":  {"workspace": "/path/handbook.workspace.json", "description": "policies, onboarding"}}}

Searching several domains: each domain returns its top hits under the chosen mode; the union is re-scored by
Defrost-Rerank (one cross-encoder, so scores are comparable across domains) and merged. merge="rrf" skips that step
and interleaves by per-domain rank instead (no reranker cost)."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from defrost_ai.config import Workspace
from defrost_ai.memory import Memory, Models
from defrost_ai.retrieval import policy


def home() -> Path:
    return Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser()


def registry_path() -> Path:
    return home() / "domains.json"


def read_registry() -> dict:
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {"domains": {}}


def register(name: str, workspace: str | Path, description: str = "") -> dict:
    reg = read_registry()
    reg["domains"][name] = {"workspace": str(Path(workspace).expanduser().resolve()), "description": description}
    registry_path().parent.mkdir(parents=True, exist_ok=True)
    registry_path().write_text(json.dumps(reg, indent=2))
    return reg


def _built_at(out: Path) -> str | None:
    try:
        return json.loads((out / "manifest.json").read_text()).get("built_at")
    except (OSError, ValueError):
        return None


class Library:
    def __init__(self, models: Models | None = None):
        self.models = models or Models()
        self._memories: dict[str, Memory] = {}
        self._lock = threading.Lock()

    def domains(self) -> dict:
        out = {}
        for name, d in read_registry()["domains"].items():
            ws = Workspace.load(d["workspace"])
            manifest = ws.out / "manifest.json"
            meta = json.loads(manifest.read_text()) if manifest.exists() else {}
            out[name] = {"description": d.get("description", ""), "workspace": d["workspace"], "out": str(ws.out),
                         "built": manifest.exists(), "built_at": meta.get("built_at"), "counts": meta.get("counts")}
        return out

    def memory(self, name: str) -> Memory:
        with self._lock:
            reg = read_registry()["domains"]
            if name not in reg:
                raise KeyError(f"unknown domain {name!r}; registered: {sorted(reg)}")
            out = Workspace.load(reg[name]["workspace"]).out
            cached = self._memories.get(name)
            built_at = _built_at(out)
            if built_at is None:                     # a build is swapping directories right now
                if cached is not None:
                    return cached                    # keep serving the previous build
                for _ in range(40):                  # first load during a swap: wait for it (<= 2 s)
                    time.sleep(0.05)
                    if (built_at := _built_at(out)) is not None:
                        break
                else:
                    raise FileNotFoundError(f"domain {name!r} has no build at {out}")
            if cached is None or cached.manifest.get("built_at") != built_at:     # reload after a rebuild
                self._memories[name] = Memory(out, self.models, name)
            return self._memories[name]

    def docs_for(self, paths: list[str], domains: list[str] | None = None) -> list[dict]:
        """Doc sections, across domains, that link to any of `paths` (code or config files). No model is loaded."""
        names = domains or [n for n, d in self.domains().items() if d["built"]]
        return [h | {"file": p} for n in names for p in paths for h in self.memory(n).docs_for(p)]

    def search(self, query: str, domains: list[str] | None = None, mode: str | None = None, k: int | str = 5,
               merge: str | None = None) -> dict:
        """mode: "accurate" (default, set in `defrost config`) or "fast"; expert modes are accepted too.
        Results of several domains are merged by the reranker in accurate mode, by rank fusion in fast mode."""
        from defrost_ai.memory import _default_mode
        mode = mode or _default_mode()
        merge = merge or ("rerank" if policy.normalize(mode) in ("accurate", "rerank", "all") else "rrf")
        names = domains or [n for n, d in self.domains().items() if d["built"]]
        if not names:
            raise RuntimeError("no built domains; run `defrost build <workspace.json> --domain NAME` first")
        qvec = self.models.retriever.embed_query(query)
        results = [self.memory(n).search(query, mode=mode, k=k, qvec=qvec) for n in names]
        if len(results) == 1:
            return results[0] | {"domains": names}
        hits = [h for r in results for h in r["hits"]]
        k = max(r["k"] for r in results)             # k="auto": the most uncertain domain decides
        if merge == "rerank":
            scores = self.models.reranker.score(query, [f"{h['heading']}\n{h['text']}" for h in hits])
            for h, s in zip(hits, scores):
                h["merge_score"] = float(s)
            hits.sort(key=lambda h: -h["merge_score"])
        else:
            hits.sort(key=lambda h: h["rank"])
        for i, h in enumerate(hits[:k]):
            h["rank"] = i + 1
        return {"query": query, "mode": mode, "domains": names, "merge": merge, "k": k, "hits": hits[:k],
                "mode_used": {n: r["mode_used"] for n, r in zip(names, results)},
                "weights_warning": next((r.get("weights_warning") for r in results if r.get("weights_warning")), None)}
