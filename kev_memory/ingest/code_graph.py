"""Code graph: graphify's AST tier (tree-sitter, deterministic, no LLM) per component, plus a Terraform/HCL reader.
Node ids get a component prefix so `index_ts` in two components stays distinct; ids stay in graphify's [a-z0-9_]
alphabet. graphify is optional (`pip install kev-memory[code]`); without it the memory is text-only."""
from __future__ import annotations

import re
from pathlib import Path

from kev_memory.config import CODE_SUFFIXES, Workspace

TF_BLOCK = re.compile(r'^\s*(resource|data|module|variable|output|locals|provider)\s*(?:"([^"]+)")?\s*(?:"([^"]+)")?\s*\{',
                      re.M)
TF_REF = re.compile(r'\b(?:(var|local|module|data)\.([A-Za-z0-9_-]+)(?:\.([A-Za-z0-9_-]+))?|(aws_[a-z0-9_]+)\.([A-Za-z0-9_-]+))')
TF_SOURCE = re.compile(r'^\s*source\s*=\s*"([^"]+)"', re.M)


def graphify_available() -> bool:
    try:
        import graphify.extract  # noqa: F401
        return True
    except ImportError:
        return False


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower()).strip("_")


def terraform(comp, root: Path, files: list[Path]):
    """HCL blocks -> nodes, references -> `references` edges, local module source -> `uses` edge. Regex-level."""
    nodes, edges, by_key = [], [], {}
    for path in files:
        text = path.read_text(errors="replace")
        rel = comp.display_path(root, path)
        fid = f"{comp.name}_{_norm(rel)}"
        nodes.append({"id": fid, "label": path.name, "file_type": "code", "source_file": rel, "source_location": None,
                      "kind": "file"})
        starts = [(m.start(), m) for m in TF_BLOCK.finditer(text)]
        for i, (pos, m) in enumerate(starts):
            kind, a, b = m.group(1), m.group(2), m.group(3)
            if kind == "locals":
                continue
            key = {"resource": f"{a}.{b}", "data": f"data.{a}.{b}", "module": f"module.{a}", "variable": f"var.{a}",
                   "output": f"output.{a}", "provider": f"provider.{a}"}[kind]
            nid = f"{comp.name}_{_norm(str(path.parent.relative_to(root)))}_{_norm(key)}"
            nodes.append({"id": nid, "label": key, "file_type": "code", "source_file": rel,
                          "source_location": f"L{text.count(chr(10), 0, pos) + 1}", "kind": f"tf_{kind}"})
            edges.append({"source": fid, "target": nid, "relation": "contains", "confidence": "EXTRACTED",
                          "confidence_score": 1.0, "source_file": rel, "weight": 1.0})
            by_key[(path.parent, key)] = nid
            body = text[pos:starts[i + 1][0] if i + 1 < len(starts) else len(text)]
            if kind == "module":
                src = TF_SOURCE.search(body)
                if src and src.group(1).startswith("."):
                    target = (path.parent / src.group(1)).resolve()
                    if target.is_dir() and root in target.parents:
                        mid = f"{comp.name}_{_norm(str(target.relative_to(root)))}_module_dir"
                        nodes.append({"id": mid, "label": f"module dir {target.relative_to(root)}", "file_type": "code",
                                      "source_file": comp.display_path(root, target), "source_location": None,
                                      "kind": "tf_module_dir"})
                        edges.append({"source": nid, "target": mid, "relation": "uses", "confidence": "EXTRACTED",
                                      "confidence_score": 1.0, "source_file": rel, "weight": 1.0})
            for r in TF_REF.finditer(body):
                ref = (f"{r.group(1)}.{r.group(2)}" if r.group(1) in ("var", "module") else
                       f"data.{r.group(2)}.{r.group(3)}" if r.group(1) == "data" else
                       f"local.{r.group(2)}" if r.group(1) == "local" else f"{r.group(4)}.{r.group(5)}")
                edges.append({"source": nid, "target": ("pending", path.parent, ref), "relation": "references",
                              "confidence": "EXTRACTED", "confidence_score": 1.0, "source_file": rel, "weight": 1.0})
    resolved = []
    for e in edges:
        if isinstance(e["target"], tuple):
            t = by_key.get((e["target"][1], e["target"][2]))
            if t is None or t == e["source"]:
                continue
            e = {**e, "target": t}
        resolved.append(e)
    return nodes, resolved


def build_code_graph(ws: Workspace, cache_dir: Path, log=print) -> dict:
    """-> {"nodes", "edges", "components": {name: stats}}; graphify's own per-file cache makes rebuilds incremental."""
    if not graphify_available():
        log("graphify not installed: code graph skipped (text-only memory). pip install 'kev-memory[code]'")
        return {"nodes": [], "edges": [], "components": {}}
    from graphify.extract import extract
    all_nodes, all_edges, stats = {}, [], {}
    for comp in ws.components:
        if comp.text_only:
            continue
        n0, e0 = len(all_nodes), len(all_edges)
        for root in comp.roots:
            files = [p for r, p in comp.files(CODE_SUFFIXES) if r == root]
            if files:
                res = extract(files, cache_root=cache_dir / comp.name)
                id_map = {n["id"]: f"{comp.name}_{n['id']}" for n in res["nodes"]}
                for n in res["nodes"]:
                    src = n.get("source_file") or ""
                    p = Path(src) if Path(src).is_absolute() else root / src
                    n = {**n, "id": id_map[n["id"]], "component": comp.name,
                         "source_file": comp.display_path(root, p) if src and root in p.resolve().parents else src}
                    all_nodes[n["id"]] = n
                for e in res["edges"]:
                    if e["source"] in id_map and e["target"] in id_map:
                        all_edges.append({**e, "source": id_map[e["source"]], "target": id_map[e["target"]]})
            tf = [p for r, p in comp.files({".tf"}) if r == root]
            if tf:
                nodes, edges = terraform(comp, root, tf)
                for n in nodes:
                    all_nodes.setdefault(n["id"], {**n, "component": comp.name})
                all_edges += edges
        stats[comp.name] = {"nodes": len(all_nodes) - n0, "edges": len(all_edges) - e0}
        log(f"  code graph {comp.name}: {stats[comp.name]}")
    return {"nodes": list(all_nodes.values()), "edges": all_edges, "components": stats}
