# End to end: stock graphify vs graphify + kev-memory, on repos the models never saw

Run on 2026-10-01. Code, questions and raw results: `benchmarks/e2e/`.

## Setup

**Repos.** Three Python projects with real docs and code: uvicorn, cattrs and structlog (commits in
`benchmarks/e2e/sources.json`). None of them is in any training set. A 13-gram scan of their docs and source
against every file our models were trained on found only shared boilerplate (encode's CONTRIBUTING text,
Tidelift notices, ASGI spec wording that starlette also uses). Every section that shares even one 13-gram with
training data was excluded from the question pool, so no gold section was seen in training. The Qwen2.5 base
model's pretraining is out of our control, and that applies equally to Claude, which runs every arm.

**Questions.** 100 paired questions (`benchmarks/e2e/questions.jsonl`, sha256 `d7cd94eb55e2…`), 53 "behaviour"
(plain words, no identifiers) and 47 "named" (one identifier or option). Each question has two golds:
- a documentation section that answers it;
- one to three code symbols that implement it.

Claude Sonnet wrote each question from one section. It picked the code symbols from the project's full symbol list,
not from kev-memory's doc→code links, so the code gold does not favour our linker. A second, independent Sonnet call
audited every item: does the section answer it, do the symbols implement it, does the question copy a phrase from
the section, is it too generic. That audit kept 100 of 117 usable items. I then hand-checked a random 15. All 15
have the right gold section and a right primary symbol. Two had a weaker secondary symbol, and one gold that was a
docstring node rather than code was removed.

**Systems.**

| system | what it is | build cost |
|---|---|---|
| graphify-ast | stock graphify 0.4.32, `graphify update` (AST only, the free tier) | 0 tokens |
| graphify-full | stock graphify 0.4.32 after the `/graphify` skill (Claude semantic tier over docs + code) | **$11.87** in Claude usage (uvicorn $4.99, cattrs $2.61, structlog $4.27) |
| kev | graphify fork + kev-memory, `memory_search` mode `fast`, k=5 | $0, 17 min local (Apple GPU, shared with other jobs) |
| kev+graph | kev context followed by the graphify-ast context | as above |

graphify is queried on the question's own repo graph, which is oracle routing in its favour. kev searches all three
repos at once.

## 1. Retrieval: what each system puts in front of the model (100 questions)

| system | gold doc in context | gold code in context | both | context tokens |
|---|---|---|---|---|
| graphify-ast | 0.00 | 0.61 | 0.00 | 1570 |
| graphify-full | 0.06 | 0.60 | 0.03 | 1567 |
| **kev** | **0.94** | 0.49 | 0.47 | 1413 |
| **kev+graph** | **0.94** | **0.80** | **0.76** | 2988 |

kev's exact gold section is in its top 5 for 92% of questions (behaviour 0.925, named 0.915).

The stock graph query (`query_graph`: label keyword match → BFS → nodes sorted by degree, cut at the budget)
almost never reaches documentation nodes, even after the paid semantic tier: the low-degree doc nodes fall
behind the budget cut. graphify's code-hit number is lenient, because its context lists up to ~180 nodes and
counts as a hit when any of them is a gold symbol. kev lists at most 6 linked symbols per section.

## 2. Answers from the context alone (RAGAS NVIDIA metrics; Sonnet answers, Haiku judges)

| system | answer accuracy | context relevance | groundedness | nv_mean | names the gold code |
|---|---|---|---|---|---|
| graphify-ast | 0.220 | 0.458 | 0.258 | 0.312 | 0.24 |
| graphify-full | 0.217 | 0.515 | 0.278 | 0.337 | 0.23 |
| **kev** | **0.895** | **0.968** | **0.973** | **0.945** | 0.50 |
| **kev+graph** | **0.895** | **0.970** | 0.922 | 0.929 | **0.60** |

- **kev vs graphify-full:** nv_mean **+0.608 [+0.554, +0.661]** (paired bootstrap, 5000 resamples).
- **What graphify's context lacks:** with graphify-full's context, the answerer said the context does not answer
  the question in 60 of 100 cases.
- **Judge sanity check:** I checked the judge output tuples for the silent-failure patterns found earlier. No
  failure pattern dominates. The common graphify tuples are (0, 0.5, 0), meaning "no answer from a partly relevant
  context".
- **Effect of adding the code graph:** it raises "names the gold code" from 0.50 to 0.60 at the cost of twice the
  tokens.

## 3. Real agents (Claude Sonnet with Read/Grep/Glob, 45 questions stratified by repo and style)

Each arm runs in its own copy of the repo and is installed the way a user would install it:
- `graphify claude install` adds the CLAUDE.md rules and the PreToolUse hook;
- the `graphify` CLI is allowed through Bash;
- the MCP server is attached.

The graphify and kev arms use the same semantic graph. They differ only in the memory tools and in the one
CLAUDE.md rule that points to them.

| arm | answer accuracy | names the gold code | cost / question | turns | time |
|---|---|---|---|---|---|
| none (file tools only) | 0.906 | 0.978 | $0.042 | 5.2 | 26 s |
| stock graphify | 0.906 | 0.933 | $0.094 | 6.2 | 43 s |
| **graphify + kev-memory** | **0.922** | 0.956 | **$0.071** | 6.9 | 36 s |

- **kev vs stock graphify:**
  - Accuracy is +0.017 [−0.044, +0.078], which is not significant.
  - Cost is **−$0.023 per question [−0.035, −0.012]**, about 25% less.
- **Tool use:** the kev agents called `memory_search` in 45 of 45 runs.

## What this shows, and what it does not

- **As a retrieval layer, kev-memory clearly beats graphify.** On unseen repos, graphify's query returns code
  structure but almost never the documentation that answers a question; kev returns it 94% of the time. When a
  model must answer from the retrieved context (a RAG pipeline, a small model, a strict token budget), that is the
  difference between 0.34 and 0.95 nv_mean.
- **Within the graphify setup, kev-memory cuts cost.** It is 25% cheaper per question than stock graphify at equal
  accuracy, and graphify's semantic tier also costs about $4 per repo to build where kev-memory costs $0.
- **For a strong agent on small repos, neither tool adds accuracy.** Sonnet with grep and file reads finds the
  answer about 91% of the time anyway, and is cheapest with no graph tools at all. The value of the memory grows
  where grep stops working: large or multi-repo corpora, docs separate from code, private knowledge bases, weaker or
  cheaper answer models, and fixed context budgets. Those cases are the next thing to measure.
- **Limits.** n = 100 (context) and 45 (agents). The questions are LLM-written and LLM-audited, with a 15-item
  human spot check. The agent eval has a ceiling effect, and the judge is an LLM.
