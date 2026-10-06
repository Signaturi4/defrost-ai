---
type: reference
entity: agent QA results
owner: maintainers
status: current
updated: 2026-10-05
lifecycle: living
source: true
---

# Agent QA results

Each study here ran `bench/agent_qa/aqa.py` on a real project. The project's questions and logs stay with the
project; this page keeps the numbers, the setup and the root causes. The method is in
[`bench/agent_qa/README.md`](../bench/agent_qa/README.md).

Facts:
- This page → records → agent QA studies of defrost on real projects
- Questions and raw logs → stay → with each project

## Study v0: old docs without defrost against cleaned docs with defrost (2026-10-05)

The project is a private ETL and CRM repository with 1,447 code files and 72 doc files. In 5 days its docs drifted
from the code: three owner decisions cancelled work that the docs still described as built.

| Item | Setting |
|---|---|
| Arm `before/plain` | Docs before the restructure, own `CLAUDE.md`; no MCP servers, no hooks, no skills |
| Arm `clean/defrost` | Docs after the restructure and a content pass, own `CLAUDE.md`, defrost memory (MCP + prompt hook, doc trust `low`), user skills including `extraction-ready-docs` |
| System version | defrost 1.2.1, weights 1.1.0 (Defrost-Ret-B, Defrost-Rerank v2), built from `feature/monitoring` `875c9a7`; retrieval code as on `main`. Skill `knowledge_lifecycle_skill` `02bb4a2` |
| Questions | 20: 6 stale traps, 3 conflict traps, 3 navigation, 2 decision, 6 fact |
| Runs | 3 repeats × 20 × 2 = 120, `claude-sonnet-5` effort medium; judge `claude-opus-5-5`, blind |
| Cost | Answers $6.55, judge $3.94 |

Q07 is left out: its ground truth expected an old log line, while both trees mark another task `doing`.

| Metric | `before/plain` | `clean/defrost` |
|---|---|---|
| Correctness, mean 0–2 | 1.61 | 1.79 |
| Fully correct | 65% | 79% |
| Wrong (0) | 4% | 0% |
| Conflict flagged on conflict traps | 0% | 44% |
| Stale fact stated as current | 0% | 0% |
| Hallucination | 7% | 7% |
| Answered in 1 turn | 21% | 39% |
| Latency, median / p90 | 11 s / 20 s | 9 s / 21 s |
| Cost per run, mean | $0.049 | $0.060 |

Paired by question and repeat, `clean/defrost` scored higher 12 times, lower 2 times and the same 43 times
(sign test p = 0.013). The judge agreed with 10 hand scores on correctness 6 times (it was stricter every time) and
on the other fields 9 or 10 times.

Facts:
- `clean/defrost` → answered → 79% fully correct, against 65% for `before/plain`
- The defrost `search` tool → was called → 0 times in 60 runs; the prompt hook carried the memory
- Conflict traps → were flagged → in 44% of `clean/defrost` answers

## Findings from v0

- **The prompt hook does all the work.** The agent never called `search` or a skill. A question the hook misses
  gets no memory.
- **Prompt text steers the hook.** With an instruction preamble in the prompt, the top hit for a spend-cap
  question was the installed rules page `KNOWLEDGE_RULES.md`, not project content.
- **A documented gap hides a conflict.** A policy page that described a code gap led the agent to call the
  disagreement "a known bug", not a conflict.
- **Stale docs did not mislead this model.** Both arms checked the code on every stale trap.
- **Extra context costs money.** The memory text and 141 skill descriptions make each run about 22% dearer.

Root causes per failed answer and the improvement loop follow in study v1.

Facts:
- Installed rule pages → compete with → project docs in search
- A documented code gap → suppressed → conflict flags

## Study v1: research loop on the prompt hook (2026-10-06)

Same project and questions as v0. Two fixes to the measurement came first: the judge now scores against key points
per question (it had marked identical answers differently), and two ground truths were wrong (the code had moved
past the docs the truths were taken from). Re-judged, v0 reads 77% fully correct without defrost and 95% with it.

Each failed v0 answer got one root cause. Most were not retrieval misses: 6 were wrong ground truths, 4 came from a
doc that framed a code gap as settled, 3 from stopping before reading the code, 2 from overclaiming, and 1 from the
judge. The hook's fast top 3 held an essential section for 15 of 20 questions, and both sides of a conflict for 0 of 3.

Every change was tested alone, cheapest test first: gold-span recall (free), the hook's injected text (free), then
targeted agent runs. A change was kept only if its metric rose and correctness did not fall.

| Change | Measured | Result | Kept |
|---|---|---|---|
| Hook: reranked search, top 5, 3,500-token pack | essential section injected; both conflict sides injected | 15/20 → 16/20; 0/3 → 3/3 | yes |
| Hook: report every disagreement with both versions; no partial yes/no | 8 hardest questions × 2: fully correct, conflicts flagged | 94% → 100%; 67% → 100% | yes |
| Hook: "verified only for code you read" sentence | 20 questions × 2 | no gain; removing it raised 1-turn answers 12% → 20% | removed |
| Hook: file:line code lines for the identifiers a section names | 20 questions × 2 | fully correct 100% → 98%, flagged 100% → 83% | no |
| Hook: "read all needed files in one step" | 20 questions × 2 | turns unchanged, fully correct 100% → 98% | no |
| Index `.plans`-like hidden folders; skip the rules page | 6 affected questions × 2 | fully correct 100% → 92% | no |
| Dense retriever Qwen3-Embedding-0.6B instead of Ret-B | gold-span recall and MRR, fast and reranked | 15/20 and .706 against 15/20 and .688; same reranked | no |

Final configuration on all 20 questions × 2 runs (`claude-sonnet-5`, judge `claude-opus-5-5`):

| Metric | Clean docs, no defrost | Defrost v0 hook | Defrost v1 hook |
|---|---|---|---|
| Fully correct | 80% | 95% (v0 study, 60 runs) | 100% |
| Conflict traps flagged | 17% | 44% (v0 study) | 100% |
| Hallucination | 0% | 0% | 0% |
| Answered in 1 turn | 22% | 37% | 20% |
| Latency, median | 9 s | 9 s | 15 s |
| Cost per run, mean | $0.051 | $0.061 | $0.102 |

The v1 hook buys correctness and conflict reporting with time and money: the agent now checks the code it is told
disagrees. A held-out set (22 questions over this project and defrost itself, written by a separate agent and never
seen by the loop) is the next validation; until it runs, these numbers are from the questions the loop was tuned on.

Facts:
- The reranked top-5 hook → put → both sides of every conflict trap in context (3/3, was 0/3)
- The v1 hook → reached → 100% fully correct and 100% conflicts flagged on the dev questions
- Code lines in the hook, folder indexing and Qwen3-Embedding-0.6B → were rejected → no measured gain

## Sources

- Study v0 and v1 runs, 2026-10-05 and 2026-10-06: `bench/agent_qa/aqa.py` on a private ETL and CRM repository;
  raw logs, scores and the per-run root-cause analysis are kept with that project.
- Loop results log (`results.tsv`, one row per experiment, kept or not), 2026-10-06, same location.
