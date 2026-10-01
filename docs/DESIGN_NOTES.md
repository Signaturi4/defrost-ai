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

## Embedding model in use

The embedding model is **Kev-Ret-B**, trained for this project:

- **Base:** Qwen2.5-0.5B (revision `060db649`), a causal model.
- **Converted to an encoder** so every token sees the whole text, in two LoRA stages:
  - MNTP (masked next-token prediction) on 50M characters of technical prose;
  - CGSA (contrastive sentence alignment).
- **Retrieval training:** a third LoRA stage (rank 16) trained on 57k question–passage pairs (MS MARCO, NQ,
  HotpotQA, StackExchange and others, plus 9.7k tech-doc questions).
- **Output:** 896-dimensional vectors, mean-pooled and normalised, up to 512 tokens.
- **Input format:** documents are embedded as "heading path + text"; queries get a fixed instruction prefix that
  is left out of the pooling.

Kev-Ret-B works next to BM25 (SQLite FTS5). **Kev-Rerank** (same backbone + LoRA + a score head, a cross-encoder)
reorders the candidates when BM25 and Kev-Ret-B disagree on the top section (the `fast` policy). The shipped version is
**Kev-Rerank v2** (weights v1.1.0). It was trained with harder negatives: other sections of the same file, and
changelog / release-note sections that mention the same names. It also added long-prose questions. On the locked
test it beats v1 by +0.024 nDCG@10 in `fast` mode ([RESULTS.md](RESULTS.md) §2.2).

Can a bigger backbone be dropped in? No. The adapters are shaped for the 0.5B model (hidden size 896, 24 layers),
while Qwen2.5-1.5B has hidden size 1536 and 28 layers, so they do not load on it. Without our training, neither
backbone retrieves well: on held-out dev, raw 0.5B scores 0.132 nDCG@10 and raw 1.5B scores 0.027, against 0.885 for
Kev-Ret-B. A 1.5B version means retraining all four stages.
