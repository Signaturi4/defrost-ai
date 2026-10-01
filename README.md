# defrost-ai

A local knowledge memory for codebases and document collections. You ask a question in plain words and get back
the documentation sections that answer it, quoted with file and line numbers, together with the code each section
names. Agents use it through MCP or the CLI; nothing leaves your machine and there is no per-query API cost.

```sh
$ kev-memory search "how do I bind values to a structlog logger so they show up in every message?" -k auto
mode fast -> rerank

[1] e2e:structlog/docs/processors.md:L39-90  processors.md > Processors > Chains > Examples
### Examples If you set up your logger like: structlog.configure(processors=[f1, f2, f3])
log = structlog.get_logger().bind(x=42) and call log.info("some_event", y=23), it results in …
```

It is a working research release. The Python package and CLI are called `kev-memory` and the models are
**Kev-Ret-B** and **Kev-Rerank**; the project itself is **defrost-ai**: your project's memory, unfrozen. It is rebuilt
from the docs and code on every commit instead of going stale in a wiki.

## Why

Code graphs such as [graphify](https://github.com/safishamsi/graphify) are good at structure: which function calls
which, what lives in which module. They are weak at the question developers ask most: *"how do I…"*, *"why
does…"*, *"what happens when…"*. The answer to those is usually a paragraph in the docs, and a keyword query over
node labels rarely reaches it. On three repos our models never saw, graphify's query put the answering doc section
in its context for **6%** of questions; this memory did for **94%** (details below).

The usual fix is a hosted embedding API or an LLM pass over every document, which costs money on every refresh and
sends your code out. This project trains small models instead (a 0.5B-parameter backbone) that run on a laptop.

## Use cases

- **Coding agents (Claude Code, Codex, Cursor…):** `memory_search` as an MCP tool next to graphify's graph tools.
  The agent gets the doc section and the code it names in one call.
- **RAG over internal docs:** handbooks, runbooks, ADRs, READMEs across many repos, with citations to path and lines.
- **Company knowledge base, several domains at once:** register each repo or folder as a domain; one query searches
  all of them and merges the results with the reranker.
- **Always fresh:** graphify's git hooks refresh the memory after every commit and branch switch. Only changed
  sections are re-embedded, and the previous build is kept for rollback.
- **Reviewable updates and memory-grounded agents:** two [Shepherd](https://github.com/shepherd-agents/shepherd)
  tasks keep a memory refresh as a reviewable result (accept, or discard and roll back) and answer questions from cited
  context in a sandbox.

## Built on

| component | origin | used for |
|---|---|---|
| **Qwen2.5-0.5B** (rev `060db649`) | Alibaba Qwen, Apache-2.0 | the backbone of both models |
| **MNTP + CGSA** recipe (KG-BiLM / LLM2Vec) | McGill NLP, MIT (`training/source/kg_bilm_experiments`) | turning the causal decoder into a bidirectional text encoder: masked next-token prediction, then contrastive sentence alignment |
| **Kev-Ret-B** (ours) | LoRA r16 on the backbone, contrastive training on 57k (query, passage, hard negative) rows: MS MARCO, NQ, HotpotQA, AllNLI, Quora, StackExchange + 9.7k tech-doc questions | dense retrieval of doc sections |
| **Kev-Rerank v2** (ours) | same backbone + LoRA + score head, listwise loss over 1 positive + 7 negatives (BM25, same-file siblings, changelog sections) | reordering the top 40 candidates |
| **SQLite FTS5 BM25** | SQLite | keyword retrieval: exact identifiers, flags, error strings |
| **`fast` policy** (ours) | no parameters | uses the cheap fusion when BM25 and the dense retriever agree on the top section, the reranker when they disagree (about half the queries) |
| **adaptive k** (ours, optional `k="auto"`) | temperature-scaled dense confidence | sends 1–5 sections: 18% fewer context tokens at the same hit rate |
| **graphify 0.4.32** | safishamsi/graphify, MIT | tree-sitter AST code graph, git hooks, MCP server, CLI. We add a patch (`integrations/graphify/`): `graphify memory …`, three MCP tools, the hook call, and a CLAUDE.md rule |
| **Shepherd** (`shepherd-ai`) | shepherd-agents | sandboxed agent tasks with retained, reviewable outputs |
| **RAGAS 0.4.3** (NVIDIA metrics) | explodinggradients/ragas | answer-level evaluation: accuracy, context relevance, groundedness |

What is different from the parts it is built on:
- **graphify** indexes code structure and, in its paid semantic tier, uses Claude to extract concepts from docs.
  This project indexes every doc section locally, links each one to the exact code nodes it names (from graphify's
  own AST graph), and ranks sections with trained models. It reuses graphify's graph, hooks and MCP server rather
  than replacing them.
- **Off-the-shelf embedders** (bge-small and similar) are trained on web text. Kev-Ret-B starts from a backbone
  adapted to technical prose and is trained on developer questions about documentation. Same-sized rerankers
  trained on web data scored lower on our held-out repos (0.717 for bge-reranker-base vs 0.890 for Kev-Rerank v2).

## Results

All numbers are nDCG@10 or RAGAS NVIDIA metrics, with paired-bootstrap 95% CIs. Every choice was made on dev
splits, and the test splits were scored once. A 13-gram leakage gate separates all training data from every suite.
Full protocol: [docs/EVALUATION.md](docs/EVALUATION.md). Everything that worked and did not:
[docs/RESULTS.md](docs/RESULTS.md).

**Locked test, against a plain BM25 section index** (175 questions; held-out OSS repos + a private 7-repo product):

| | held-out repos (106) | private product repos (69) |
|---|---|---|
| nDCG@10, BM25 → `fast` | 0.703 → **0.840** | 0.684 → **0.835** |
| RAGAS nv_mean (both), BM25 → `fast` (Kev-Rerank v1) | 0.791 → **0.879** | |

Kev-Rerank v2 (default since weights v1.1.0) against v1 on the same locked test, paired: `fast` +0.024
[+0.010, +0.041], `rerank` +0.037 [+0.019, +0.056]. The RAGAS row and the graphify comparison below were run with
v1; they were not re-run.

**End to end vs stock graphify, on three repos never used in training** (uvicorn, cattrs, structlog; 100 paired
doc + code questions; [docs/E2E_GRAPHIFY.md](docs/E2E_GRAPHIFY.md)):

| | stock graphify (paid semantic tier) | graphify + this memory |
|---|---|---|
| answering doc section in the context | 6% | **94%** |
| answer quality from that context (RAGAS nv_mean) | 0.337 | **0.945** (+0.61 [+0.55, +0.66]) |
| build cost for the 3 repos | $11.87 of Claude usage | $0, runs locally |
| Claude Sonnet agent with file tools: accuracy | 0.906 | 0.922 (n.s.) |
| Claude Sonnet agent: cost per question | $0.094 | **$0.071** (−25%, CI excludes 0) |

The honest caveat: on repos this small, a strong agent with plain grep also scores 0.906 and is the cheapest arm.
The memory matters most where grep stops working: large or multi-repo corpora, docs kept apart from code, weaker or
cheaper answer models, and fixed context budgets.

**Why not a pure knowledge graph?** The graph is reliable for structure and doc→code links, not as the place
answers come from: instructions keep their conditions and exceptions in paragraphs. See
[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md), which also describes the embedding model.

**Known weaknesses:** the dense retriever alone loses to BM25 on private product docs; the reranker takes 6–9 s
per query on Apple GPU; a 568M public reranker still beats ours on long narrative prose (books 0.949 vs 0.902);
multi-hop questions are unsolved.

## Quick start

One command installs the CLI, downloads the weights and registers the Claude Code integration for all projects:

```sh
curl -fsSL https://raw.githubusercontent.com/Signaturi4/defrost-ai/main/install.sh | sh
```

It needs [uv](https://docs.astral.sh/uv/). It installs `kev-memory` as a uv tool (Python 3.12), fetches the
adapters (about 140 MB, checked by sha256) to `~/.cache/kev-memory`, and runs `kev-memory claude install --user`.

Then open Claude Code in any repository and type:

```
/defrost-setup
```

Claude asks how the memory should stay fresh and builds it. The default is the recommended one: **build now, then
refresh on every merge or commit to main that changes a doc or code file**. You never run a build by hand.

| refresh option | what it installs | when it runs |
|---|---|---|
| Build now + on every merge/commit to main (Recommended) | `post-merge` + `post-commit` git hooks (marked block, your own hooks are kept) | after a merge or commit on `main`/`master`, only if an indexed file changed |
| Every few hours (2 / 6 / 24 h) | a launchd agent on macOS, a crontab line on Linux | on the schedule, only if something changed |
| When a Claude session starts | a `SessionStart` hook in `.claude/settings.json` | at session start, in the background, only if stale |
| Only when I run `/memory-update` | nothing | on demand |

Every trigger runs `kev-memory refresh <domain> --if-changed` in the background, so commits and sessions do
not wait. Refreshes are incremental (only new or edited sections are re-embedded) and log to
`~/.kev-memory/<domain>.refresh.log`. The memory lives in `~/.kev-memory/<domain>`, never in your repo.
Indexing follows `.gitignore`, so ignored folders (archives, data dumps, `node_modules`) stay out.
Setup also adds a short "Project memory" block to `CLAUDE.md` that tells Claude to call `memory_search` for
how/why questions. Without it, agents in our evals mostly ignored the MCP tools.

**Doc trust.** `/defrost-setup` asks how far Claude should trust your docs; `--doc-trust` sets it:

| setting | for | what Claude does with a hit |
|---|---|---|
| `low` (default; "code is the truth") | startups, code that changes daily, few docs | treats the section as a hint, reads the `verify in:` files, answers from the code and lists doc/code conflicts |
| `high` ("docs are reliable") | legacy or well-documented projects | answers from the section; reads code only when a hit carries a `!` stale or conflict line |

The setting lives in `~/.kev-memory/<domain>.workspace.json` (`"doc_trust"`). Every search result starts with
a line that states it, and the `CLAUDE.md` block says the same. Re-run `kev-memory setup . --doc-trust high
--build skip` to change it; no rebuild is needed. The default is `low` because a wrong answer copied from a stale
doc is silent, while grounding costs a few file reads.

The same setup without Claude:

```sh
kev-memory setup . --build now --on-main-merge          # + --doc-trust high|low, --every-hours 6, --claude-hook, --doc-rules
kev-memory status                                       # per domain: built_at, counts, stale (and why), triggers
kev-memory refresh my-repo --if-changed                 # what the triggers run
kev-memory setup . --remove-triggers                    # remove hooks, schedule and session hook
```

Upgrade: run the install line again (it reinstalls; the resident service restarts itself on the new build).

## Use with Claude (MCP server + slash commands)

`install.sh` already did this for all projects. To do it for one project only (commands in `.claude/commands`,
server in `.mcp.json`, so the team gets it from git):

```sh
kev-memory claude install .
```

Or register the server alone, in any MCP client (Claude Code, Claude Desktop, Cursor):

```sh
claude mcp add defrost -- kev-memory mcp
```

| in Claude | what it does |
|---|---|
| `/defrost-setup` | asks how to keep the memory fresh, installs those triggers, builds the memory |
| `/memory-search <question>` | searches, then answers from the returned sections with `path:Lstart-end` citations and the linked code |
| `/memory-update [domain]` | incremental refresh now |
| `/memory-init [name]` | builds a memory for a repo or docs folder, no triggers |
| `/memory-domains` | lists the memories with counts and build times |

MCP tools: `memory_search` (modes `fast`, `rerank`, `dense`, `bm25`, …; `k` = number or `"auto"`),
`memory_domains`, `memory_init`, `memory_update`, `memory_job`, `memory_rollback`. The server is a thin stdio
process with no ML dependencies. Searches go to one resident local service (`kev-memory serve`, started on first
use), so the models load once for all clients. Set `KEV_MEMORY_DOMAINS=my-repo` to restrict searches by default.

### With graphify

graphify users can get the same search inside graphify's own CLI and MCP server (patch on v0.4.32):

```sh
git clone https://github.com/safishamsi/graphify && cd graphify && git checkout v0.4.32 \
  && git apply ../defrost-ai/integrations/graphify/graphify-0.4.32-kev-memory.patch && pip install -e ".[mcp]"
export KEV_MEMORY_SERVE_CMD="kev-memory serve"
graphify memory init . --domain my-repo && graphify claude install
graphify memory search "how are refunds issued?" -k auto
```

Python:

```python
from kev_memory import Library
res = Library().search("how does the feed reach mobile", mode="fast", k="auto")
for hit in res["hits"]:
    print(hit["domain"], hit["path"], hit["lines"], hit["heading"], [c["label"] for c in hit["code"]])
```

Modes: `fast` (default), `rerank` (most accurate, slowest), `dense`, `bm25` (exact strings), `hybrid`, `all`.

**Grounded in the code, not just the docs.** Docs go stale, so every hit carries the files to check it against:

```
[2] my-repo:docs/DEVOPS.md:L51-64  DEVOPS.md > Known problem: the CI deploy never runs
...
  -> code deploy.yml (my-repo/.github/workflows/deploy.yml)
  ! doc may be stale: my-repo/.github/workflows/deploy.yml changed 2026-09-29, after this doc (2026-09-26)
  ! doc/code conflict: names not found in the code: `scripts/old_deploy.sh`
  verify in: my-repo/.github/workflows/deploy.yml
```

- **What gets linked:** config, CI and infra files (compose files, Dockerfiles, workflows, crontabs, shell scripts),
  as well as code symbols. Secret-like files (`.env`, `*secret*`, keys) are never indexed.
- **Stale docs:** a hit is flagged when a file it names was committed after the doc was.
- **Conflicts:** a hit is flagged when the doc names a file or function that no longer exists. The agent does not
  pick a side: it shows the doc and the code and asks you ("code is right", "doc is right", "not a conflict", "not
  sure"). Your answer is recorded (`memory_resolve_conflict`, `kev-memory conflicts <domain>`) and shown on later
  hits as a `resolved:` line, so each conflict is asked once.
- **Claude's instructions:** the CLAUDE.md block tells Claude to read the `verify in:` files before stating how
  something behaves, to trust the code when the two disagree, and to list the doc/code conflicts it found.
- **Docs to update:** after a change, `memory_docs_for` (MCP) or `kev-memory docs-for <file>` lists the doc sections
  that describe the changed files, so they are updated in the same change.

FAQ / support chatbots: see [docs/FAQ_CHATBOT.md](docs/FAQ_CHATBOT.md) (use `hybrid` for real-time, about 45 ms).

## Writing docs the memory reads well

`/defrost-setup` offers this kit, or run `kev-memory setup . --doc-rules` (the kit is
`kev_memory/assets/doc_rules/`, also linked as `templates/doc-rules/`). It appends a short, highlighted rule block to the end of `CLAUDE.md` and installs:
- the full rules, read only when docs are written, so they don't fill every session;
- a glossary template;
- a linter;
- a Facts extractor that turns `- Subject → relation → Object` lines into triples.

The research behind the rules: [docs/WRITING_FOR_EXTRACTION.md](docs/WRITING_FOR_EXTRACTION.md).

## Weights

The LoRA adapters (MNTP, CGSA, Kev-Ret-B, Kev-Rerank v2 + score head, about 140 MB) are attached to the
[v1.1.0 release](https://github.com/Signaturi4/defrost-ai/releases/tag/v1.1.0). `install.sh` (or
`kev-memory download-weights`, or the first search) fetches them to `~/.cache/kev-memory/models` and checks the
archive's sha256. By hand:

```sh
curl -L -o weights.tar.gz https://github.com/Signaturi4/defrost-ai/releases/download/v1.1.0/defrost-ai-weights-v1.1.0.tar.gz
tar xzf weights.tar.gz                   # -> models/  (sha256 of the archive: 494fab8997ce01c4…)
kev-memory verify-weights                # checks every file against models/MANIFEST.json
```

They load on top of `Qwen/Qwen2.5-0.5B` at revision `060db649` (downloaded from Hugging Face on first use).
You can also keep them elsewhere and set `KEV_MEMORY_MODELS`. `training/` holds the exact scripts and configs
used to train them, stage by stage.

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

Expected on held-out dev (nDCG@10): bm25 0.711, dense 0.885, hybrid 0.789, rerank 0.890, fast 0.887
(weights v1.1.0; with v1.0.0: rerank 0.870, fast 0.870).

## License

Code: MIT. Model adapters: LoRA weights on Qwen2.5-0.5B (Apache-2.0). `training/source/kg_bilm_experiments`: MIT
(McGill NLP). The benchmark books keep their own licenses (Pro Git CC BY-NC-SA 3.0, Eloquent JavaScript CC BY-NC,
500 Lines or Less CC BY 3.0); only questions and line references are included, not the texts.
