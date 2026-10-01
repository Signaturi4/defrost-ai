# Context offload and doc sync: agents keep their state and the docs in step with the code

Status: prototype on branch `feature/context-offload` (2026-10-02). Unit tests pass. The A/B eval below is planned,
not run.

## The problem

A long Claude Code session carries its whole history: tool outputs, dead ends, superseded plans. That costs tokens
on every turn, and the model can lose the thread. Our agent eval showed memory search is cheaper than grep for one
question. This extends the idea to the session itself: write the working state into the memory, start a fresh
context, and give it only the goal and the task context.

## What Claude Code offers (and what actually shrinks context)

Sources: code.claude.com docs (hooks, hooks-guide, how-claude-code-works, glossary), via Context7.

| mechanism | effect on context | notes |
|---|---|---|
| `/clear` | **removes** the history | a `SessionStart` hook with matcher `clear` fires afterwards; its stdout is added to the new context |
| `/compact [focus]`, auto-compaction | **replaces** history with a model-written summary (lossy) | "Compact Instructions" in CLAUDE.md steer it; a `SessionStart` hook with matcher `compact` fires after |
| `PreCompact` hook | none | it only receives `trigger` and `custom_instructions`; it cannot write or change the summary |
| subagents | **isolate** work; only the result comes back | good for searches and long reads |
| MCP tools | definitions deferred by default; output stays in context | only names and server instructions are loaded up front |
| CLAUDE.md, imports, auto memory | **add** to every request | survive compaction; keep them short |
| `UserPromptSubmit` additional context | **adds** on every turn | not used here, to avoid adding text on every turn |

No hook can replace the history in the middle of a session. Only `/clear` and `/compact` shrink it. So the design
makes `/clear` safe: the state goes into a note first, and the hook brings back only that note.

## Design

```
 long session ──/handoff──► remember(kind="note", goal, state, decisions, next_steps, files)
                              │ writes <project>/defrost-memory/notes/<time>-<slug>.md (one git commit)
                              │ indexes it incrementally as domain "<domain>-context"
              ──/clear────► SessionStart hook (matcher clear|compact): `defrost hook brief`
                              │ prints goal + state + next steps + files of the newest note (≤ 350 words)
 fresh session ◄────────────┘ older decisions: search(q) covers "<domain>-context" by default
```

Decisions:
- **Notes live outside the project's git**, in the context repository `defrost-memory/` (its own git repo, excluded
  from the project's; see [LETTA_CONTEXT_REPOS.md](LETTA_CONTEXT_REPOS.md)). They hold transient session state:
  half-made decisions, failed attempts, personal constraints. That should not land in the project's history or docs
  domain. Decisions that last belong in the real docs (the doc rules already ask for that). The notes are a
  separate domain, so searches can include or exclude them.
- **One note is one page, and every section names the goal.** Each section makes sense alone in search results,
  following the same rules as `docs/WRITING_FOR_EXTRACTION.md`. The sections are Goal, State of the work,
  Decisions, Next steps and Files.
- **The brief is capped and model-free.** `defrost note --brief` (the hook: `defrost hook brief`) reads one file: no torch, about 0.04 s. It works as a
  hook even when the service is down.
- **The hook runs on `clear` and `compact`.** After an auto-compaction, the structured note sits next to the lossy
  summary as an anchor. It does not fire at startup or on resume, so a normal new session is unchanged.
- **Opt-in, project-level only.** It is part of the `standard` and `full` setup profiles. That writes the hook into the
  project's `.claude/settings.json`, never into `~/.claude/settings.json`. `defrost setup --remove` removes it.
- **The model writes the note.** `/handoff` asks for about 300 words in a fixed shape. That is the same trust as
  `/compact`, but the output is structured, searchable, and kept across sessions.

Pieces:

| piece | where |
|---|---|
| note writing, brief, notes domain, hook install | `defrost_ai/notes.py` |
| MCP tool `remember(kind="note")` | `defrost_ai/service/mcp_server.py` |
| CLI `defrost note`, `defrost note --brief`, `defrost hook brief` | `defrost_ai/cli.py`, `project_setup.py` |
| slash command `/handoff` | `defrost_ai/integrations/claude_commands/handoff.md` |
| tests (no weights) | `tests/test_notes.py` |

Checked end to end on an isolated home and port: a note was written and indexed, and `search` on the
notes domain returned the Decisions section first. The brief printed in 0.04 s.

## Doc sync: document every change, after edits and on commit

The memory is only as good as the docs. Agents change code and leave the docs behind; the next memory search then
repeats the stale doc. Doc sync closes that loop with a model-free planner and three hooks.

**Planner** (`defrost_ai/docsync.py`, `defrost docs`, MCP `docs_for`; 0.2–0.4 s on general_crm).
From a diff (the working tree, the staged changes, or one commit) it lists:

| list | how it is found |
|---|---|
| sections to update | the section links to a changed symbol (the symbol's span overlaps a changed hunk), or it names identifiers from the changed lines |
| also linked | sections linked to a changed file only; shown as a count per file |
| undocumented files | new or changed code with no linked section |
| new names | env vars, CLI flags and `script:task` names on added lines that no doc section mentions |
| already covered | doc files edited in the same change; their sections are not asked again |

Links are checked before they are trusted. A path-like mention must be a suffix of the changed path, because
graphify node ids collide for files with the same name. A bare lowercase word (`proxy`, `next`) is ignored.

**Triggers** (`standard` profile: `post-commit` and `SessionStart`; `full` adds `Stop` and `PreToolUse`; project `.claude/settings.json` and the git `post-commit`):

| moment | hook | effect |
|---|---|---|
| agent finishes editing | `Stop` → `defrost hook stop` | blocks the stop once per change set with the plan and the `/document-changes` procedure |
| agent runs `git commit` | `PreToolUse` (Bash) → `defrost hook commit` | denies the commit once per staged change set; the agent updates and stages the docs, then commits again |
| you commit outside Claude | git `post-commit` → `defrost hook post-commit` | records the plan as a pending task (< 1 s, no model) |
| next Claude session | `SessionStart` (startup, resume) → `defrost hook pending` | lists pending commits; Claude offers `/document-changes <sha>` |
| optional: right after a commit outside Claude | `--docs-auto` | background `claude -p "/document-changes <sha>"`; edits docs only, never commits, `--max-budget-usd 0.5` per commit; skipped inside Claude (`$CLAUDECODE`) |

Each gate asks at most once per change set and never blocks twice in a row (`stop_hook_active`), so the agent
cannot loop. A broken hook exits 0 and never blocks work.

**`/document-changes [sha | --staged]`** is the doc counterpart of `/handoff`:
1. Plan with `defrost docs` (MCP `docs_for`), then read the diff.
2. Read the code before the docs; update only what the change made false.
3. Document new commands, env vars and files in the page that already covers that area.
4. Follow `docs/DOC_RULES.md`, the rules from `WRITING_FOR_EXTRACTION.md`.
5. Mark doc/code contradictions as open questions instead of guessing.
6. Run `doc_lint`.
7. Resolve the pending task and write a handoff note.

**Checked on a clone of general_crm** (read-only use of its built memory):
- **Real commit `ae68e9b6` (beta gate):** the plan named `api.md` "Gates in `proxy.ts`" and DEPLOY_ENCORE ↔
  `deploy/start-api.sh`. It listed 6 new files with no doc, and the new names `access-code:ensure`,
  `BETA_ACCESS_CODE` and `BETA_GATE`. These are exactly the gaps both agents missed in the earlier memory-vs-grep
  comparison.
- **New env var:** a one-line edit adding `MIGRATE_ON_START` was blocked once at stop and once at commit, then
  allowed.
- **Commit from the terminal:** a pending task was recorded in 0.7 s and shown by the SessionStart hook.
- **`--docs-auto` dry run:** printed the capped command. It was not run.

### Doc-sync evaluation (not run: needs your go-ahead)

- **Tasks:** 10 recent general_crm commits whose docs were updated later or never. Rebuild the memory at each
  parent commit, then run `/document-changes <sha>` with Sonnet on a scratch clone.
- **Metrics:**
  - Recall of the doc sections a human later changed, from git history.
  - Precision: sections edited that the change did not affect.
  - `doc_lint` errors.
  - Contradictions flagged.
  - Cost per commit.
- **Size:** about 10 × $0.30 ≈ **$3**, plus a 30-minute hand review of the 10 diffs.

## Limits

- **Automatic compaction does not write a note.** `PreCompact` cannot call the model. The user, or a CLAUDE.md
  rule ("write a handoff before the context gets long"), has to trigger `/handoff`.
- **A bad note loses state.** If the model leaves something out, the fresh session cannot know it. The eval below
  measures how often that happens.
- **`/clear` drops the prompt cache.** The first turns after it are not cached, but they are much smaller.

## Evaluation plan (not run: needs your go-ahead, it costs Claude tokens)

**Question.** After a handoff, does a fresh session finish the task as well as continuing the full conversation,
and with how many fewer tokens?

**Setup.**
- **Repos:** 10 two-phase tasks on scratch copies of the e2e repos (uvicorn, cattrs, structlog), each with a test
  that decides success. Never on client repos.
- **Phase 1** runs once per task: Sonnet does the first half (investigation plus a partial change) with
  `claude -p`, so every arm starts from the same state.
- **Phase 2** is run three ways:

| arm | how phase 2 starts |
|---|---|
| A: full history | `claude -p --resume <phase-1 session>` |
| B: handoff | `/handoff` in the phase-1 session, then a fresh `claude -p` in the repo with the hook installed |
| C: compact | `/compact` in the phase-1 session, then continue |

**Metrics.**
- **Primary:** phase-2 success, meaning the task's test passes.
- **Secondary:** phase-2 input and output tokens, turns, cost, wall time, repeated dead ends (a phase-1 failure tried
  again), and constraint violations (a user rule from phase 1 broken in phase 2).
- **Report:** paired per task, with a bootstrap CI on the token difference.

**Pass criteria (set before the run).** Arm B's success is at least arm A's minus one task out of 10, and B's
phase-2 input tokens are at least 40% below A's.

**Size and cost (estimate).**

| item | calls | estimated cost |
|---|---|---|
| phase 1 | 10 × about $0.25 | ≈ $2.5 |
| handoff notes | 10 × about $0.03 | ≈ $0.3 |
| phase 2, 3 arms × 10 | 30 × about $0.20 | ≈ $6 |
| **total, one repeat** | about 50 agent runs, about 1.5–2.5M tokens | **≈ $9 (≈ $18 with 2 repeats)** |

The plan also needs a 20-minute hand check of the 10 task definitions before any run.
