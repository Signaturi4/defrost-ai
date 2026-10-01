"""Build or update a memory from a workspace.

    kev-memory build workspace.json          # first build and every later update: the same command

Steps: code graph (graphify AST, cached per file) -> documents -> sections -> doc->code links -> Kev-Ret-B vectors.
Updates are incremental where it costs: section vectors are cached by the sha256 of the embedded text, so only new
or edited sections are encoded; graphify caches its per-file extraction. The SQLite index is rewritten each time
(seconds). The new build is written to a staging directory and swapped in atomically, so readers never see a
half-built memory; the previous build is kept as <out>.previous for rollback."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev_memory import store
from kev_memory.config import DOC_SUFFIXES, Workspace, git_head, git_times
from kev_memory.ingest.code_graph import build_code_graph
from kev_memory.ingest.documents import document_id, split_sections
from kev_memory.ingest.links import core_paths, link_section, symbol_index
from kev_memory.models.encoder import ENCODER_ID


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    c = np.load(path)
    return dict(zip(c["keys"], c["vecs"]))


def file_times(ws: Workspace, nodes: list[dict]) -> dict[str, int]:
    """{display path: last commit unix time} for every file a code or config node lives in (one git log per root)."""
    out = {}
    for comp in ws.components:
        for root in comp.roots:
            prefix = comp.display_path(root, root / "_")[:-1]
            paths = {n["source_file"]: root / n["source_file"][len(prefix):] for n in nodes
                     if n.get("component") == comp.name and (n.get("source_file") or "").startswith(prefix)}
            paths = {k: v for k, v in paths.items() if v.exists()}
            times = git_times(root, list(paths.values())) if paths else {}
            out.update({k: times[v] for k, v in paths.items() if v in times})
    return out


def build(workspace: str | Path, models=None, log=print) -> dict:
    ws = Workspace.load(workspace)
    out = ws.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + ".building")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    cache_root = store.cache_dir(out)                        # persists across builds, outside the swapped directory
    cache_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    old = json.loads((out / store.MANIFEST).read_text()) if (out / store.MANIFEST).exists() else {}

    log(f"[1/4] code graph ({ws.name})")
    code = build_code_graph(ws, cache_root / "graphify", log)
    code["file_times"] = file_times(ws, code["nodes"])          # last commit per code/config file: staleness check
    (stage / store.CODE_GRAPH).write_text(json.dumps(code))
    idx = symbol_index(code["nodes"], code["edges"])
    core = core_paths(ws, code["nodes"])
    component_of = {n["id"]: n["component"] for n in code["nodes"]}

    log("[2/4] documents -> sections -> links")
    db = store.create(stage / store.KNOWLEDGE_DB)
    doc_hashes, per_component = {}, defaultdict(lambda: defaultdict(int))
    for comp in ws.components:
        files = list(comp.files(DOC_SUFFIXES))
        times = {}
        for root in comp.roots:
            times.update(git_times(root, [p for r, p in files if r == root]))
        for root, path in files:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            rel = comp.display_path(root, path)
            doc = document_id(comp.name, rel)
            sections = split_sections(text, path.suffix)
            title = next((p[0] for _, p, *_ in sections if p), path.stem)
            digest = hashlib.sha256(raw).hexdigest()
            doc_hashes[rel] = digest
            db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)",
                       (doc, comp.name, rel, str(path), digest, times.get(path, 0), len(text.split()), title))
            stats = per_component[comp.name]
            stats["docs"] += 1
            for k, (level, heading, a, b, body) in enumerate(sections):
                sid = f"{doc}_s{k}"
                links, unresolved, n_mentions, n_resolved = link_section(body, idx, component_of, comp.name, core)
                db.executemany("INSERT INTO links VALUES (?,?,?,?,?)", [(sid, *l) for l in links])
                db.executemany("INSERT INTO unresolved VALUES (?,?)", [(sid, m) for m in unresolved])
                db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (sid, doc, comp.name, rel, " > ".join([path.name] + heading), level, k, a, b, body,
                            len(body.split()), n_mentions, n_resolved, len(unresolved)))
                stats["sections"] += 1
                stats["links"] += sum(1 for l in links if l[2] == "EXTRACTED")
        for p in comp.skipped:
            log(f"  skipped (iCloud file not downloaded): {p}")
    db.execute("INSERT INTO sections_fts(sections_fts) VALUES ('rebuild')")
    db.commit()

    log("[3/4] section vectors (Kev-Ret-B, cached by text hash)")
    rows = db.execute("SELECT id, heading_path, text FROM sections ORDER BY rowid").fetchall()
    texts = [f"{h}\n{t}" for _, h, t in rows]
    keys = [_sha(t) for t in texts]
    cache = _load_cache(cache_root / "embeddings.npz")
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        from kev_memory.memory import Models
        models = models or Models()
        by_key = {k: t for k, t in zip(keys, texts)}
        vecs = models.retriever.embed_documents([by_key[k] for k in missing])
        cache.update({k: v.astype(np.float16) for k, v in zip(missing, vecs)})
    dim = next(iter(cache.values())).shape[0] if cache else 896
    section_vecs = np.stack([cache[k] for k in keys]).astype(np.float16) if keys else np.zeros((0, dim), np.float16)
    np.savez(stage / store.SECTION_VECTORS, ids=np.array([r[0] for r in rows]), vecs=section_vecs,
             encoder=np.array(ENCODER_ID))
    live = set(keys)
    kept = {k: v for k, v in cache.items() if k in live}
    np.savez(cache_root / "embeddings.npz", keys=np.array(list(kept)), vecs=np.stack(list(kept.values())) if kept
             else np.zeros((0, dim), np.float16))

    log("[4/4] manifest + atomic swap")
    old_hashes = old.get("doc_hashes", {})
    changed = {"added": sorted(set(doc_hashes) - set(old_hashes)), "removed": sorted(set(old_hashes) - set(doc_hashes)),
               "edited": sorted(p for p in doc_hashes if p in old_hashes and old_hashes[p] != doc_hashes[p])}
    manifest = {
        "workspace": ws.name, "workspace_file": str(ws.source), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "encoder": ENCODER_ID, "build_seconds": round(time.time() - t0, 1),
        "counts": {"docs": len(doc_hashes), "sections": len(rows), "code_nodes": len(code["nodes"]),
                   "code_edges": len(code["edges"]), "sections_encoded_now": len(missing),
                   "doc_code_links": int(db.execute("SELECT COUNT(*) FROM links WHERE confidence='EXTRACTED'").fetchone()[0])},
        "components": {c: dict(s) for c, s in per_component.items()},
        "sources": {str(r): git_head(r) for c in ws.components for r in c.roots},
        "core": core,
        "changed": {k: v[:200] for k, v in changed.items()} | {"n": {k: len(v) for k, v in changed.items()}},
        "doc_hashes": doc_hashes,
    }
    (stage / store.MANIFEST).write_text(json.dumps(manifest, indent=1))
    db.close()
    previous = out.with_name(out.name + ".previous")
    if out.exists():
        if previous.exists():
            shutil.rmtree(previous)
        out.rename(previous)
    stage.rename(out)
    log(f"done: {manifest['counts']} in {manifest['build_seconds']}s; changed docs {manifest['changed']['n']}")
    return manifest


def rollback(workspace: str | Path) -> bool:
    """Swap <out> and <out>.previous (undo the last build)."""
    out = Workspace.load(workspace).out
    previous = out.with_name(out.name + ".previous")
    if not previous.exists():
        return False
    tmp = out.with_name(out.name + ".swap")
    out.rename(tmp)
    previous.rename(out)
    tmp.rename(previous)
    return True
