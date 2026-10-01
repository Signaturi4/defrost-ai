# Architecture: graphify + kev-memory + Shepherd

The rule: build on graphify and Shepherd, and reinvent nothing they already do. kev-memory contributes the model,
meaning the trained retriever and reranker plus the index they search. The rest is glue.

```
                ┌──────────────────────── graphify (fork: ~/graphify-kev, branch kev-memory) ─────────────────────┐
  code  ──────► │ AST extraction (tree-sitter) ─► graph.json / GRAPH_REPORT.md      graphify hook install         │
                │ MCP server: query_graph, get_node, … + memory_search, memory_update, memory_domains             │
                │ CLI: graphify query / path / explain … + graphify memory init|update|search|domains|rollback    │
                └───────────────┬──────────────────────────────────────────────────────────▲─────────────────────┘
                                │ extract() per component (same AST, same cache)          │ HTTP (stdlib client)
  docs  ──────► ┌───────────────▼──────────────────────────────── kev-memory ──────────────┴─────────────────────┐
                │ builder: sections (md/rst/adoc) ─► doc→code links ─► Kev-Ret-B vectors (cached by text hash)   │
                │ retrieval: BM25 (FTS5) + Kev-Ret-B dense ─► fast policy ─► Kev-Rerank on disagreement          │
                │ library: many domains, cross-domain merge by Kev-Rerank      service: resident models (HTTP)  │
                └───────────────▲─────────────────────────────────────────────────────────────────────────────────┘
                                │ client.search / client.update
                ┌───────────────┴──────────────────── Shepherd (shepherd-ai) ─────────────────────────────────────┐
                │ refresh DOMAIN  -> memory snapshot as a retained output (select = accept, reject = rollback)    │
                │ ask "question"  -> cited context -> sandboxed Claude agent -> answer as a retained output       │
                └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Who does what

| need | provided by | kev-memory adds |
|---|---|---|
| code structure (AST nodes, edges) | graphify `extract()` | nothing; calls it per component and prefixes ids |
| agent tools (MCP) | graphify `serve.py` | three tools registered in graphify's server |
| refresh on commit / branch switch | graphify `hook install` (post-commit, post-checkout) | one call after graphify's own rebuild, gated by `graphify-out/kev-memory.json` |
| CLI | graphify | `graphify memory ...` subcommand |
| sandboxed agents, reviewable results, revert | Shepherd tasks + retained outputs | two tasks: `record_memory_snapshot`, `answer_from_memory` |
| text retrieval | **kev-memory** (replaces graphify's token-costing semantic tier for docs) | Kev-Ret-B + Kev-Rerank + BM25, `fast` policy |

## Why a resident service

The models are torch (about 1 GB in memory). graphify and Shepherd stay dependency-light and talk to one local
service (`kev-memory serve`, default port 8765). The service starts on first use, keeps the models loaded, reloads a
domain when its manifest changes, and runs one build at a time (one GPU lock).

## Integration code

- graphify changes: `integrations/graphify/graphify-0.4.32-kev-memory.patch`
  (applies to tag `v0.4.32`; +244 / −3 lines, mostly the new `graphify/memory.py`).
- Shepherd tasks: `kev_memory/integrations/shepherd.py`.

## Verified end to end (2026-09-30)

- **MCP:** graphify's server lists its 7 graph tools + 3 memory tools; `memory_search` answers through the service.
- **Hooks:** `graphify memory init` → `graphify hook install` → commit. graphify rebuilt the code graph, the hook
  queued the memory update, and the new section was searchable right after, with its doc→code link.
- **Shepherd refresh:** a snapshot was retained as run `run-2ceb1254225d` (0 sections re-encoded: cache hit).
- **Shepherd ask:** a sandboxed Claude agent wrote a cited answer from three retrieved sections, retained as run
  `run-d622a52e5f5b`, in 18 s.
- **Parity:** the package reproduces the evaluated system exactly on held-out dev (sections, rankings, nDCG@10).
