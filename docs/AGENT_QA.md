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

Root causes per failed answer and the improvement loop follow in the next study.

Facts:
- Installed rule pages → compete with → project docs in search
- A documented code gap → suppressed → conflict flags
