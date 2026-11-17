# Hindsight (vectorize-io) and defrost-ai: two kinds of agent memory

Researched 2026-11-17. Sources:
- Hindsight: https://github.com/vectorize-io/hindsight (MIT, about 61.5k stars), read at commit `a46baef`
- defrost-ai: this repository at `be65fd2` (1.3.0), [RESULTS.md](RESULTS.md), [BRAND.md](BRAND.md)
- Slides: [slides/hindsight-vs-defrost-ai-2026-11-17.pdf](slides/hindsight-vs-defrost-ai-2026-11-17.pdf), built
  with the `client-phase-proposal-deck` skill from `Signaturi4/my-skills-library` (spec in the same folder)

## 1. Different questions

| | defrost-ai | Hindsight |
|---|---|---|
| Answers | "How does this work, why is it so?" from the team's docs | "What happened, what does this user prefer?" from the agent's history |
| Stores | Doc sections with `path:Lstart-end`, links to the code they name, recorded doc/code decisions and notes | World facts, experiences, observations (consolidated beliefs with evidence and a proof count), mental models and pages |
| Write path | No LLM. Sections are split by heading and re-embedded only when they change (commit, merge, schedule) | An LLM call per `retain` extracts facts, entities, time and relations; consolidation runs in the background |
| Search | SQLite FTS5 BM25 + Defrost-Ret-B; Defrost-Rerank v2 only when the two disagree (`accurate`) | Vector, BM25, graph and temporal retrieval in parallel, reciprocal rank fusion, cross-encoder rerank |
| Reasoning | None in the memory; the agent reasons, guided by doc trust and `verify in:` lines | `reflect`: an LLM reasons over the memories, shaped by bank disposition traits |
| Infra | CLI and MCP server, SQLite, about 141 MB of adapters, offline | API server with Postgres + pgvector (or embedded pg0, Oracle), Docker, Helm, or Hindsight Cloud |
| Claude Code | `defrost claude install`: MCP server, hooks (prompt context, handoff), slash commands, `/defrost-setup` | MCP endpoint per bank; `hindsight-coding-agents` builds a per-repo bank from git history and past sessions |
| Privacy and cost | Nothing leaves the machine; $0 build | Depends on the LLM provider; tokens on every retain and reflect (local providers possible) |
| Evidence | Held-out repos with paired-bootstrap CIs: 94% vs 6% answering section in context against stock graphify; nDCG@10 0.840 vs BM25 0.703 | LongMemEval 94.6% and LoCoMo 92.0%, self-reported; LongMemEval reproduced by Virginia Tech's Sanghani Center (per the README) |

The benchmarks measure different jobs (doc retrieval against long conversational memory), so no number in one
column can be compared with the other.

## 2. When to use which

- **defrost-ai:** large or multi-repo docs, code that changes daily, client code that must stay local, no token budget.
- **Hindsight:** assistants that serve users over many sessions, learning from outcomes, a shared memory server.
- **Both:** they do not overlap in data. Two MCP servers in one agent: defrost-ai for how and why, Hindsight for who
  and when.

## 3. Ideas defrost-ai can borrow

| Idea in Hindsight | defrost-ai version | Size |
|---|---|---|
| Observations keep quotes and a proof count, refined rather than overwritten | Recorded doc/code decisions keep their evidence (both sides, `path:line`) and a count of later confirmations | S |
| Mental models and pages: standing answers rewritten in the background | A page per frequent question, refreshed after merges, read without search | M |
| Memory Defense: secret and PII scan on every retain | Scan `remember()` input with the gitleaks patterns; redact or refuse | S |
| Coding-agent ingest of git history and past sessions | Opt-in, local summary of past handoff notes into the context repo | M |
| Webhooks and Prometheus metrics | A refresh-finished event and counters for `defrost status` | S |
| A live benchmarks site | One results page per release, same protocol ([EVALUATION.md](EVALUATION.md)) | S |

Keep what differs: local and free, cited sections, doc/code checks by doc trust, held-out evaluation, and the plain
statement that grep wins on small repos.
