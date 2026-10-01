Here is the revised README. The Quick Start has been moved immediately below the intro and reorganized logically so a new user downloads the weights first, installs the package, and then chooses their integration path (Claude or Graphify). The deeper architectural and evaluation details have been pushed down.

---

# defrost-ai

A local knowledge memory for codebases and document collections. You ask a question in plain words and get back the documentation sections that answer it, quoted with file and line numbers, together with the code each section names. Agents use it through MCP or the CLI; nothing leaves your machine and there is no per-query API cost.

```sh
$ kev-memory search "how do I bind values to a structlog logger so they show up in every message?" -k auto
mode fast -> rerank

[1] e2e:structlog/docs/processors.md:L39-90  processors.md > Processors > Chains > Examples
### Examples If you set up your logger like: structlog.configure(processors=[f1, f2, f3])
log = structlog.get_logger().bind(x=42) and call log.info("some_event", y=23), it results in …

```

It is a working research release. The Python package and CLI are called `kev-memory` and the models are **Kev-Ret-B** and **Kev-Rerank**; the project itself is **defrost-ai**: your project's memory, unfrozen. It is rebuilt from the docs and code on every commit instead of going stale in a wiki.

---

## 🚀 Quick Start

To use `defrost-ai`, you need to download the model adapters and install the package. Then, you can integrate it into your preferred workflow.

### 1. Download Weights & Install

The LoRA adapters load on top of Qwen2.5-0.5B (downloaded automatically on first use).

```sh
# Download and extract the model weights
curl -L -o weights.tar.gz https://github.com/Signaturi4/defrost-ai/releases/download/v1.0.0/defrost-ai-weights-v1.0.0.tar.gz
tar xzf weights.tar.gz                   # Creates the models/ directory

# Install the package with MCP and code analysis dependencies
pip install -e ".[code,mcp]"

# Verify the weights downloaded correctly
kev-memory verify-weights                

```

### 2. Choose Your Integration

**Option A: Use with Claude (Easiest)**
Register the MCP server and install four slash commands (`/memory-init`, `/memory-search`, `/memory-update`, `/memory-domains`) for Claude Code:

```sh
kev-memory claude install .          # Installs .claude/commands + .mcp.json

```

*Alternatively, register the server in any MCP client (Claude Desktop, Cursor) with:* `claude mcp add defrost -- kev-memory mcp`

**Option B: Use with Graphify (Advanced Code Hooks)**
Integrate with `graphify` to refresh memory automatically on every commit and branch switch.

```sh
# Clone and patch graphify
git clone https://github.com/safishamsi/graphify && cd graphify && git checkout v0.4.32 \
  && git apply ../defrost-ai/integrations/graphify/graphify-0.4.32-kev-memory.patch 
pip install -e ".[mcp]"

export KEV_MEMORY_SERVE_CMD="kev-memory serve"

# Build and search memory in your repo
cd ~/src/my-repo
graphify memory init . --domain my-repo  
graphify hook install                    
graphify memory search "how are refunds issued?" -k auto

```

---

## Why

Code graphs such as [graphify](https://github.com/safishamsi/graphify) are good at structure: which function calls which, what lives in which module. They are weak at the question developers ask most: *"how do I…"*, *"why does…"*, *"what happens when…"*. The answer to those is usually a paragraph in the docs, and a keyword query over node labels rarely reaches it. On three repos our models never saw, graphify's query put the answering doc section in its context for **6%** of questions; this memory did for **94%** (details below).

The usual fix is a hosted embedding API or an LLM pass over every document, which costs money on every refresh and sends your code out. This project trains small models instead (a 0.5B-parameter backbone) that run on a laptop.

## Use cases

* **Coding agents (Claude Code, Codex, Cursor…):** `memory_search` as an MCP tool next to graphify's graph tools. The agent gets the doc section and the code it names in one call.
* **RAG over internal docs:** handbooks, runbooks, ADRs, READMEs across many repos, with citations to path and lines.
* **Company knowledge base, several domains at once:** register each repo or folder as a domain; one query searches all of them and merges the results with the reranker.
* **Always fresh:** graphify's git hooks refresh the memory after every commit and branch switch. Only changed sections are re-embedded, and the previous build is kept for rollback.
* **Reviewable updates and memory-grounded agents:** two [Shepherd](https://github.com/shepherd-agents/shepherd) tasks keep a memory refresh as a reviewable result (accept, or discard and roll back) and answer questions from cited context in a sandbox.

## Built on

| component | origin | used for |
| --- | --- | --- |
| **Qwen2.5-0.5B** (rev `060db649`) | Alibaba Qwen, Apache-2.0 | the backbone of both models |
| **MNTP + CGSA** recipe (KG-BiLM / LLM2Vec) | McGill NLP, MIT (`training/source/kg_bilm_experiments`) | turning the causal decoder into a bidirectional text encoder: masked next-token prediction, then contrastive sentence alignment |
| **Kev-Ret-B** (ours) | LoRA r16 on the backbone, contrastive training on 57k (query, passage, hard negative) rows: MS MARCO, NQ, HotpotQA, AllNLI, Quora, StackExchange + 9.7k tech-doc questions | dense retrieval of doc sections |
| **Kev-Rerank** (ours) | same backbone + LoRA + score head, listwise loss over 1 positive + 7 BM25 negatives | reordering the top 40 candidates |
| **SQLite FTS5 BM25** | SQLite | keyword retrieval: exact identifiers, flags, error strings |
| **`fast` policy** (ours) | no parameters | uses the cheap fusion when BM25 and the dense retriever agree on the top section, the reranker when they disagree (about half the queries) |
| **adaptive k** (ours, optional `k="auto"`) | temperature-scaled dense confidence | sends 1–5 sections: 18% fewer context tokens at the same hit rate |
| **graphify 0.4.32** | safishamsi/graphify, MIT | tree-sitter AST code graph, git hooks, MCP server, CLI. We add a patch (`integrations/graphify/`): `graphify memory …`, three MCP tools, the hook call, and a CLAUDE.md rule |
| **Shepherd** (`shepherd-ai`) | shepherd-agents | sandboxed agent tasks with retained, reviewable outputs |
| **RAGAS 0.4.3** (NVIDIA metrics) | explodinggradients/ragas | answer-level evaluation: accuracy, context relevance, groundedness |

What is different from the parts it is built on:

* **graphify** indexes code structure and, in its paid semantic tier, uses Claude to extract concepts from docs. This project indexes every doc section locally, links each one to the exact code nodes it names (from graphify's own AST graph), and ranks sections with trained models. It reuses graphify's graph, hooks and MCP server rather than replacing them.
* **Off-the-shelf embedders** (bge-small and similar) are trained on web text. Kev-Ret-B starts from a backbone adapted to technical prose and is trained on developer questions about documentation. Same-sized rerankers trained on web data scored lower on our held-out repos (0.717 for bge-reranker-base vs 0.870).

## Results

All numbers are nDCG@10 or RAGAS NVIDIA metrics, with paired-bootstrap 95% CIs. Every choice was made on dev splits, and the test splits were scored once. A 13-gram leakage gate separates all training data from every suite. Full protocol: [docs/EVALUATION.md](https://www.google.com/search?q=docs/EVALUATION.md). Everything that worked and did not: [docs/RESULTS.md](https://www.google.com/search?q=docs/RESULTS.md).

**Locked test, against a plain BM25 section index** (175 questions; held-out OSS repos + a private 7-repo product):

|  | held-out repos (106) | private product repos (69) |
| --- | --- | --- |
| nDCG@10, BM25 → `fast` | 0.703 → **0.814** | 0.684 → **0.814** |
| RAGAS nv_mean (both), BM25 → `fast` | 0.791 → **0.879** |  |

**End to end vs stock graphify, on three repos never used in training** (uvicorn, cattrs, structlog; 100 paired doc + code questions; [docs/E2E_GRAPHIFY.md](https://www.google.com/search?q=docs/E2E_GRAPHIFY.md)):

|  | stock graphify (paid semantic tier) | graphify + this memory |
| --- | --- | --- |
| answering doc section in the context | 6% | **94%** |
| answer quality from that context (RAGAS nv_mean) | 0.337 | **0.945** (+0.61 [+0.55, +0.66]) |
| build cost for the 3 repos | $11.87 of Claude usage | $0, runs locally |
| Claude Sonnet agent with file tools: accuracy | 0.906 | 0.922 (n.s.) |
| Claude Sonnet agent: cost per question | $0.094 | **$0.071** (−25%, CI excludes 0) |

The honest caveat: on repos this small, a strong agent with plain grep also scores 0.906 and is the cheapest arm. The memory matters most where grep stops working: large or multi-repo corpora, docs kept apart from code, weaker or cheaper answer models, and fixed context budgets.

**Why not a pure knowledge graph?** The graph is reliable for structure and doc→code links, not as the place answers come from: instructions keep their conditions and exceptions in paragraphs. See [docs/DESIGN_NOTES.md](https://www.google.com/search?q=docs/DESIGN_NOTES.md), which also describes the embedding model.

**Known weaknesses:** the dense retriever alone loses to BM25 on private product docs; the reranker takes 6–9 s per query on Apple GPU; a 568M public reranker beats ours on long narrative prose (books 0.949 vs 0.862); multi-hop questions are unsolved.

## Writing docs the memory reads well

`templates/doc-rules/` is a drop-in kit for any project. `bash templates/doc-rules/install.sh /path/to/project` appends a short, highlighted rule block to the end of `CLAUDE.md` and installs:

* the full rules, read only when docs are written, so they don't fill every session;
* a glossary template;
* a linter;
* a Facts extractor that turns `- Subject → relation → Object` lines into triples.

The research behind the rules: [docs/WRITING_FOR_EXTRACTION.md](https://www.google.com/search?q=docs/WRITING_FOR_EXTRACTION.md).

## Layout

```
kev_memory/        library: ingest (sections, code graph, doc->code links), models, retrieval, builder, service, CLI
integrations/      the graphify patch
benchmarks/        frozen question suites (held-out repos, books, e2e) + the e2e harness
training/          training scripts and configs (MNTP -> CGSA -> Kev-Ret-B / Kev-Rerank), Kaggle notebooks
scripts/           parity check, benchmark source fetcher, weight export, reranker efficiency
docs/              EVALUATION, RESULTS, ARCHITECTURE, E2E_GRAPHIFY, DESIGN_NOTES, WRITING_FOR_EXTRACTION
templates/         doc-rules kit for CLAUDE.md (rules, glossary, linter, Facts extractor)

```

## Reproduce

```sh
python scripts/fetch_benchmark_sources.py          # pinned commits of the benchmark repos and books
kev-memory build benchmarks/heldout/workspace.json
kev-memory benchmark --suite benchmarks/heldout/questions.jsonl --memory ~/.kev-memory/benchmark-heldout --split dev

```

Expected on held-out dev (nDCG@10): bm25 0.711, dense 0.885, hybrid 0.789, rerank 0.870, fast 0.870.

## License

Code: MIT. Model adapters: LoRA weights on Qwen2.5-0.5B (Apache-2.0). `training/source/kg_bilm_experiments`: MIT (McGill NLP). The benchmark books keep their own licenses (Pro Git CC BY-NC-SA 3.0, Eloquent JavaScript CC BY-NC, 500 Lines or Less CC BY 3.0); only questions and line references are included, not the texts.
