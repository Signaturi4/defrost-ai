---
type: explanation
entity: retrieval tuning backlog
owner: maintainers
status: draft
updated: 2026-10-07
lifecycle: living
source: true
---

# Retrieval tuning backlog after the v3 held-out run

The v3 hook and retrieval did not carry over to a third repository (fully correct 44% -> 47%, conflict traps 0% in
both arms; see `docs/AGENT_QA.md`). Its questions were then opened as dev data and traced. This page records the
root cause and the planned changes; nothing here is implemented yet.

Facts:
- v3 → did not carry over → to a third repository's held-out questions
- This page → plans → the next tuning iteration

## Root cause: near-duplicate files fill the top 5

The third repository ships one copy of its agent skill per tool (`skill.md`, `skill-aider.md`, `skill-codex.md`,
`skill-kiro.md` and more) and its README in about 15 languages under `docs/translations/`. For 10 of 12 questions,
the five injected sections were copies of one text: five skill variants or five translations. The README and
`ARCHITECTURE.md` lines that disagree with the code were not in the pack for any of the 5 trap questions, so the
agent answered from the code and reported "no conflict". The agent did write the "Docs vs code" list (30 of 36
answers); the list only covers what was injected.

| Question kind | Doc line that disagrees injected | Pack content |
|---|---|---|
| Conflict traps (3) | 0/3 | skill copies or translations |
| Stale traps (3) | 1/3 | skill copies, one correct pack |
| Navigation, decision, fact (6) | n/a | mostly translations |

No indexing bug: all 58 docs are indexed, including the README, `ARCHITECTURE.md` and `SECURITY.md`. Where the
disagreeing sections rank (top 50 per retriever, final after reranking):

| Trap | Section | BM25 | Dense | Final |
|---|---|---|---|---|
| 1 | README, 325 words | 11 | – | 19 |
| 2 | ARCHITECTURE schema, 42 words | – | 1 | 14 |
| 3 | ARCHITECTURE module table, 202 words | 45 | – | – |
| 4 | same table | 21 | – | – |
| 5 | SECURITY table, 246 words | 24 | – | – |

Trap 2 shows the duplicate effect directly: dense search ranks the right section first, and the reranker puts 13
copies of one skill text above it. Traps 3 to 5 point at table rows, which both retrievers match poorly; a table
row is a weak unit for search (see D5 below).

Facts:
- Near-duplicate sections → crowded out → the doc lines that disagree with the code
- The "Docs vs code" list → covers → only the injected sections

## Planned changes, each with its own isolated test

| Id | Change | Cheapest test |
|---|---|---|
| D1 | Collapse near-duplicate hits: keep the best of sections whose text is near-identical (normalised text hash, or cosine > 0.95 between section vectors) and fill the freed slots | gold recall@5 on the third repository; no loss on the two dev sets |
| D2 | One section per heading path across files with the same base name pattern (`skill-*.md`), unless the question names the file | same |
| D3 | Down-weight translations: detect language per doc at build time, prefer the language of the question | same, plus non-English questions as a check |
| D5 | Index each row of a doc table as its own small unit, linked to its section | rank of table-row claims; recall@5 |
| D4 | `setup` suggests excluding generated copies and translations when it finds many near-identical files | count of near-identical files per repository |

Gate as before: keep a change only if its metric rises and correctness does not fall. Then run the frozen held-out
set on a fourth repository, which a separate agent writes before tuning starts.

Facts:
- Collapsing near-duplicates → is → the first change to test
- A fourth repository → holds → the next held-out set

## Sources

- Held-out run on the third repository, 12 questions × 3 runs × 2 arms, 2026-10-07; hook output per question
  captured with `bench/agent_qa/aqa.py hooks`.
