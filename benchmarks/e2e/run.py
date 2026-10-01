"""Retrieval contexts for every system on the e2e questions, plus doc / code hit rates.

Systems (each gets the same ~2000-token context budget, except defrost+graph which is the two concatenated):
  graphify-ast    stock graphify 0.4.32, AST graph (free tier), MCP query_graph logic (BFS depth 3)
  graphify-full   stock graphify 0.4.32 after the /graphify skill (Claude semantic tier over docs + code)
  defrost             graphify fork + defrost: memory_search mode=fast, k=5 (doc sections + linked code)
  defrost+graph       defrost context followed by the graphify-ast context (what an agent with the fork's MCP sees)
graphify is queried on the question's own repo graph (oracle routing, in its favour); defrost searches all three repos.

    .venv/bin/python run.py            # -> contexts.jsonl, hits.json"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BUDGET = 2000


def graph_ctx(G, question):
    from graphify.serve import _bfs, _score_nodes, _subgraph_to_text
    terms = [t.lower() for t in question.split() if len(t) > 2]       # exactly _tool_query_graph
    scored = _score_nodes(G, terms)
    start = [nid for _, nid in scored[:3]]
    if not start:
        return "No matching nodes found."
    nodes, edges = _bfs(G, start, 3)
    return f"Start: {[G.nodes[n].get('label', n) for n in start]} | {len(nodes)} nodes\n\n" + \
        _subgraph_to_text(G, nodes, edges, BUDGET)


def load_graph(path):
    import networkx as nx
    from networkx.readwrite import json_graph
    d = json.load(open(path))
    try:
        return json_graph.node_link_graph(d, edges="links")
    except TypeError:
        return json_graph.node_link_graph(d)


def rel(path, comp):
    return path[len(comp) + 1:] if path.startswith(comp + "/") else path


def doc_hit(ctx, q):
    return rel(q["gold_doc"]["path"], q["component"]) in ctx


def code_hit(ctx, q):
    for g in q["gold_code"]:
        lab = g["label"].strip(".").replace("()", "")
        if lab in ctx and rel(g["source_file"], q["component"]).rsplit("/", 1)[-1] in ctx:
            return True
    return False


def main():
    from defrost_ai.library import Library
    from defrost_ai.memory import Memory
    qs = [json.loads(l) for l in open(HERE / "questions.jsonl")]
    graphs = {}
    for kind, root in (("graphify-ast", HERE / "graphify"), ("graphify-full", HERE / "graphify-full")):
        for c in ("uvicorn", "cattrs", "structlog"):
            f = root / c / "graphify-out/graph.json"
            if f.exists():
                graphs[(kind, c)] = load_graph(f)
    lib = Library()
    rows = []
    for i, q in enumerate(qs):
        ctx = {}
        for kind in ("graphify-ast", "graphify-full"):
            if (kind, q["component"]) in graphs:
                ctx[kind] = graph_ctx(graphs[(kind, q["component"])], q["question"])
        res = lib.search(q["question"], ["e2e"], mode="fast", k=5)
        ctx["defrost"] = Memory.context(res, BUDGET)
        ctx["defrost+graph"] = ctx["defrost"] + "\n\n## code graph\n" + ctx["graphify-ast"]
        sec_hit = any(h["section_id"] == q["gold_doc"]["section_id"] for h in res["hits"])
        rows.append({"id": q["id"], "component": q["component"], "style": q["style"], "ctx": ctx,
                     "defrost_section_hit": sec_hit, "defrost_mode_used": res["mode_used"],
                     "doc_hit": {k: doc_hit(v, q) for k, v in ctx.items()},
                     "code_hit": {k: code_hit(v, q) for k, v in ctx.items()},
                     "tokens": {k: len(v) // 4 for k, v in ctx.items()}})
        print(f"\r{i + 1}/{len(qs)}", end="", flush=True)
    print()
    (HERE / "contexts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    systems = [s for s in ("graphify-ast", "graphify-full", "defrost", "defrost+graph") if all(s in r["ctx"] for r in rows)]
    out = {}
    for strat in ("all", "behavior", "named"):
        rs = [r for r in rows if strat == "all" or r["style"] == strat]
        out[strat] = {s: {"n": len(rs), "doc_hit": float(np.mean([r["doc_hit"][s] for r in rs])),
                          "code_hit": float(np.mean([r["code_hit"][s] for r in rs])),
                          "both": float(np.mean([r["doc_hit"][s] and r["code_hit"][s] for r in rs])),
                          "tokens": float(np.mean([r["tokens"][s] for r in rs]))} for s in systems}
        out[strat]["defrost_section_hit@5"] = float(np.mean([r["defrost_section_hit"] for r in rs]))
    (HERE / "hits.json").write_text(json.dumps(out, indent=1))
    for strat, t in out.items():
        print(f"[{strat}] n={t[systems[0]]['n']}  defrost section hit@5 {t['defrost_section_hit@5']:.3f}")
        for s in systems:
            print(f"  {s:14s} doc {t[s]['doc_hit']:.3f}  code {t[s]['code_hit']:.3f}  both {t[s]['both']:.3f}"
                  f"  tokens {t[s]['tokens']:.0f}")


if __name__ == "__main__":
    main()
