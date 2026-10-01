# Design notes: why sections, not a graph, are the source of truth

## What the research showed about the graph idea

The graph is reliable for structure and links. It was not reliable as the place answers come from. Every test
pointed the same way:

| test | result |
|---|---|
| graphify's graph query vs our section search, on unseen repos | answering doc in the context: 6% vs 94%; answer quality (RAGAS nv_mean) 0.34 vs 0.95 |
| graphify's paid semantic tier (Claude extracts doc concepts into graph nodes) | no better than its free code-only graph (0.337 vs 0.312, not significant) |
| our own model for extracting a doc knowledge graph (typed entities and relations) | F1 0.33–0.38; failed its pass threshold |
| graph neighbour expansion for multi-hop questions | no gain |

The reason: instructions and explanations live in paragraphs, with conditions, defaults and exceptions. Turning
them into nodes and edges loses that detail, and a query that matches node labels rarely reaches the right paragraph.

Where the graph clearly helps is the code side and the doc→code links. Adding graphify's code context to our
results raised "the right code is in the context" from 0.49 to 0.80.

So the design that works keeps the doc sections as the source of truth, found by hybrid search, and uses the
graph only for navigation. That is also why the Facts lines in the doc rules (`templates/doc-rules/`) are short
summaries attached to the prose, never a replacement for it.

Details: [E2E_GRAPHIFY.md](E2E_GRAPHIFY.md) (graphify comparison), [RESULTS.md](RESULTS.md) (all experiments,
including the ones that failed).

## How doc → code links are made

The build parses the code with graphify's AST extractor. Each file, class, function and method becomes a node. It
then collects code-like mentions from every doc section: backticked spans, file paths, and camelCase or snake_case
words. A mention links to a node when it names exactly one candidate. The links are stored in the memory's SQLite
file. Search never uses them for ranking (BM25 and the models rank on text alone); they are shown under each hit
(`-> code …`) and feed the `verify in:` and stale-doc lines.

A name that is unique in the code is not the same thing the doc means. We hand-labelled 100 links (general_crm
plus the held-out repos; `scripts/link_audit.py`, labels in `results/private/links/`). Precision before the
precision rules:

| mention kind | precision | typical error |
|---|---|---|
| file path | 0.94 | a user's own `wsgi.py` matched to the library's |
| code-shaped (`saveTone`, `create_environ`) | 0.96 | a JSON field that shares a function name |
| capitalised word (`Headers`) | 0.88 | `Session` (an auth model) matched to a scraper class |
| dotted, last part only (`fields.Tuple`) | 0.75 | `data.fix` (an event field) matched to `ScrapeError.fix` |
| bare lowercase word (`pending`) | 0.12 | status values, roles, columns, package names |
| target in a test file | 0.00 | `worker`, `query` matched to test helpers |

These rules now apply (`defrost_ai/ingest/links.py`):
- **Tests and examples:** test, spec and example files are link targets only through an explicit file path.
  Nested functions and builtins are never targets.
- **Methods** are indexed as `Class.method`. A bare method name only answers a code-shaped mention.
- **Lowercase words:** a bare lowercase word links only to a top-level symbol in the project core whose file name
  the same section mentions.
- **Dotted names:** a dotted name falls back to its last part only when an earlier part matches the target's
  class, file or folder.
- **Core and ordering:** with project core paths (explicit `core: [...]`, or detected server-side folders such as
  `backend/`, `api/`, `db/`, `pipeline/`, else `src/`), an ambiguous name with exactly one core candidate links
  to it. Links are shown core first, then path before qualified name before code-shaped before word.

After the rules, on the same 100 links: precision 0.60 → 0.95 (stratified sample). 37 wrong links were removed and
5 correct ones lost: two `loop.*` properties and three lowercase loop-variable names. Links the new rules added
were 15 of 16 correct. Weighted by how common each kind is, the estimated precision of shown links went from
0.80 to 0.94 on general_crm, and the estimated number of correct links rose (1,477 → 1,541), because exact
`Class.method` and checked dotted matches add links the old rules dropped as ambiguous.

## Embedding model in use

The embedding model is **Defrost-Ret-B**, trained for this project:

- **Base:** Qwen2.5-0.5B (revision `060db649`), a causal model.
- **Converted to an encoder** so every token sees the whole text, in two LoRA stages:
  - MNTP (masked next-token prediction) on 50M characters of technical prose;
  - CGSA (contrastive sentence alignment).
- **Retrieval training:** a third LoRA stage (rank 16) trained on 57k question–passage pairs (MS MARCO, NQ,
  HotpotQA, StackExchange and others, plus 9.7k tech-doc questions).
- **Output:** 896-dimensional vectors, mean-pooled and normalised, up to 512 tokens.
- **Input format:** documents are embedded as "heading path + text"; queries get a fixed instruction prefix that
  is left out of the pooling.

Defrost-Ret-B works next to BM25 (SQLite FTS5). **Defrost-Rerank** (same backbone + LoRA + a score head, a cross-encoder)
reorders the candidates when BM25 and Defrost-Ret-B disagree on the top section (the `fast` policy). The shipped version is
**Defrost-Rerank v2** (weights v1.1.0). It was trained with harder negatives: other sections of the same file, and
changelog / release-note sections that mention the same names. It also added long-prose questions. On the locked
test it beats v1 by +0.024 nDCG@10 in `fast` mode ([RESULTS.md](RESULTS.md) §2.2).

Can a bigger backbone be dropped in? No. The adapters are shaped for the 0.5B model (hidden size 896, 24 layers),
while Qwen2.5-1.5B has hidden size 1536 and 28 layers, so they do not load on it. Without our training, neither
backbone retrieves well: on held-out dev, raw 0.5B scores 0.132 nDCG@10 and raw 1.5B scores 0.027, against 0.885 for
Defrost-Ret-B. A 1.5B version means retraining all four stages.
