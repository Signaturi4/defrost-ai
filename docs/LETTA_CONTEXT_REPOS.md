# Letta Code "Context Repositories" (git-backed memory) and how defrost-ai fits with it

Researched 2026-10-02. Sources:
- Blog: https://www.letta.com/blog/context-repositories/ (2026-02-12)
- Code: https://github.com/letta-ai/letta-code (Apache-2.0), read at commit `3687ea5`

## 1. How it works

**What it stores.** The agent's own learned context: identity, user preferences, corrections, project notes and
skills. It does not index a code base. Each agent has one git repository, the MemFS, cloned to
`~/.letta/agents/<agent_id>/memory` (`src/agent/memory-filesystem.ts`). The path is exported as `$MEMORY_DIR`.

**Layout (MemFS v2, `src/agent/memory-format.ts`, `src/memory-frontmatter.ts`)**

| path | rule | in the prompt? |
|---|---|---|
| `MEMORY.md` (root) | required; no frontmatter; a map of what is *not* loaded | yes |
| other root `*.md` | core memory; frontmatter with exactly `name` + `description` | yes, every call |
| `<dir>/MEMORY.md` | each directory needs one, or its files do not count as memory | no |
| `<dir>/*.md` | deferred memory; same two-key frontmatter | no; read on demand |
| `skills/<name>/SKILL.md` | procedural memory, skill frontmatter | listed, loaded on use |

- **Progressive disclosure.** The prompt carries only the core files and the indexes. Nested files are read when an
  index line says they are relevant. Links use `[[path]]`. The guide says to keep the root under ~10% of the context
  window (15–20k tokens).
- **Limits.** A pre-commit hook enforces them (`src/agent/memory-constraints.ts`). It embeds its own validator, so it
  runs without the package installed. The defaults are:
  - `maxDepth` 2;
  - `maxFileCharacters` 20,000;
  - `maxCoreMemoryCharacters` 65,536.

  The hook also rejects bad frontmatter and protected `read_only` files. A tracked `.memfs.config.json` can override
  the limits.

**Editing memory is ordinary file work plus git.**
- The agent edits with Bash, Read, Edit and Write, then commits only the paths it changed, with an author and a
  message.
- An edit changes behaviour only after the prompt is recompiled: a new conversation, or a changed committed revision.
- The harness pushes clean commits after each turn to Letta Cloud. A custom remote can be set with
  `/memory-repository set`.

**Background memory workers** (`src/agent/subagents/memory-worker.ts`, `memory-worktree.ts`, `memory-handoff.ts`)
- **Isolation.** A `memory` subagent gets a private git worktree of the memory repo, so it never touches the main
  agent's checkout. It sees the assignment plus a transcript snapshot it can read on demand.
- **Merge back.** When the worker finishes, its commits are merged into the checkout, fast-forward when possible.
  If the merge conflicts with the main agent's own edits, the merge is aborted and the branch kept, so nothing is
  lost. A repair worker then resolves the conflict by reading both sides, never by taking one side wholesale.
- **Cancelled worker:** its worktree is discarded.
- **Reflection** (sleep-time learning, `src/reflection-settings.ts`):
  - **Triggers:** `off`, every N steps (`step-count`), or on context compaction (`compaction-event`).
  - **Merge modes:** `auto`, or `explicit`, where the agent reviews before applying.
  - **What it does:** reviews recent history and writes what is worth keeping, in a worktree.
- **Init** (`/init`, skill `initializing-memory`): subagents explore the repo, and past Claude Code and Codex histories,
  and build the first hierarchy.
- **Defragmentation:** splits, merges and re-files memory into about 15–25 focused files, after a backup.
- **Recall memory.** The full message history is stored by the server, cannot be changed, and is searched by a
  recall subagent. It is not in the MemFS.
- **Shared memory** (skill `managing-shared-memory`). Org-owned git repos, attached to several agents and mounted
  next to `$MEMORY_DIR`, using the same git workflow.

**In one line:** a small, curated, version-controlled set of markdown files that the agent rewrites itself and that
is compiled into its prompt. Retrieval is "read the index, then open the file". There is no ranking model.

## 2. How it relates to defrost-ai

| | Letta Context Repository | defrost-ai |
|---|---|---|
| holds | what the *agent* has learned (identity, preferences, corrections, notes) | what the *project* says: docs sections + linked code |
| size | small, curated (≤ ~65k chars pinned, 15–25 files) | large (hundreds to thousands of sections, multi-repo) |
| retrieval | pinned core + index files + grep | BM25 + Kev-Ret-B + Kev-Rerank, adaptive k, citations |
| writer | the agent and its memory workers | the project's own docs (plus doc sync, handoff notes) |
| freshness | every change is a commit | incremental rebuild on merge/commit/schedule |
| audit | git history of every memory edit | build manifests + rollback; conflict log (jsonl) |

They do different jobs: Letta remembers how to work with *you*, and defrost finds what the *project* says. Both can
run together, and four patterns are worth taking over.

## 3. Integration options, ranked

1. **Index the MemFS as a defrost domain** (no new code). A MemFS is a git repo of markdown with name and description
   frontmatter, which is exactly what defrost indexes.
   - Setup: `kev-memory setup ~/.letta/agents/<id>/memory --domain letta-<agent> --on-main-merge`. Our post-commit
     hook refreshes the index after each memory commit, in a few seconds.
   - Benefit: ranked semantic search over deferred memory once it outgrows "read the index". Letta's
     defragmentation exists because index-plus-grep stops scaling.
   - Usage: Letta Code supports stdio MCP servers (`src/mcp-client.ts`), so `kev-memory mcp` can be attached there.
2. **Make defrost's own working memory a context repository.** Today, handoff notes
   (`~/.kev-memory/<domain>-notes/`) and conflict decisions (`<domain>.conflicts.jsonl`) are plain files.
   - Proposal: one git repo per domain, `~/.kev-memory/<domain>-context/`, in Letta's v2 layout:
     - `MEMORY.md` as the index;
     - `handoffs/` and `decisions/conflicts/`, each file with name and description frontmatter;
     - one commit per handoff or decision.
   - Benefits:
     - Every decision becomes auditable, with `git log` and `git blame` on why a doc was declared wrong.
     - It syncs to a team remote, so conflict decisions become shared knowledge.
     - Letta agents can attach it as shared memory.
   - Cost: small. `notes.py` and `conflicts.py` gain a commit step and frontmatter; search keeps working
     because the folder is indexed as before.
3. **Run doc-sync workers in git worktrees.** `--docs-auto` currently runs `claude -p` against the live checkout.
   - Letta's pattern instead:
     - the worker edits docs in a private worktree on a branch (`defrost/docs-<sha>`) and commits there;
     - the harness merges it with fast-forward only, or leaves the branch or opens a PR when it conflicts;
     - a cancelled worker's worktree is discarded.
   - This fits the human-in-the-loop conflict rule. It mirrors Letta's `explicit` merge mode, where the
     user approves the branch before it lands. It also removes the risk of a background run touching files you
     are editing.
4. **Validate docs in a pre-commit hook, the way Letta validates memory.**
   - What it checks: run the doc-rules linter (`docs/tools/doc_lint.py`) and simple limits (section size, required
     Facts lines) in a pre-commit hook. Make the validator self-contained, as Letta embeds its own, so it works
     without the package.
   - Why: the rules then hold for humans and agents alike, and the retrieval quality the rules buy does not decay.
5. **Write the handoff automatically on compaction**, mirroring Letta's `compaction-event` reflection trigger.
   - How: a Claude Code `PreCompact` hook receives the transcript path. It can start a background, budget-capped
     `claude -p` (Haiku) that writes the handoff note, committed in a worktree as in option 2.
   - Limit: this closes the gap noted in CONTEXT_OFFLOAD.md, where no note is written before an auto-compaction.
     It spends Claude tokens, so it must stay opt-in.

**What not to copy:**
- **Pinning project knowledge into the prompt.** Letta's core memory is meant for behaviour-shaping rules, and the
  measurements in this repo show that answers live in sections found on demand.
- **The Letta server dependency.** MemFS v2 assumes a Letta backend (cloud or local) for agent state, recall and
  sync. The file format and git workflow can be adopted without it.

## 4. What we adopted (branch `feature/context-repo`)

Letta Code's logic was ported to Python, not vendored: the TypeScript is not shipped. Attribution is in `NOTICE`,
the Apache-2.0 license is in `third_party/letta-code/LICENSE`, and every ported module names its sources in its
header.

| defrost-ai module | ported from (Letta Code, commit 3687ea51) | what changed |
|---|---|---|
| `kev_memory/context_constraints.py` | `src/memory-frontmatter.ts`, `src/memory-constraints.ts`, `src/agent/memory-constraints.ts` | one layout (memfs v2); allowed keys are `name`, `description` and the protected `read_only`; config edits need `DEFROST_CONTEXT_CONFIG_UPDATE=1` |
| `kev_memory/context_repo.py` | `src/agent/memory-format.ts`, `src/agent/memory-git-hooks.ts`, `src/agent/memory-scanner.ts`, maintenance skills | one repository per project domain, not per agent; indexes are generated from frontmatter; sync is a plain git remote, with no Letta server |
| `kev_memory/worktree.py` | `src/agent/memory-worktree.ts`, `src/agent/memory-operation.ts`, `src/utils/worktree-lock.ts` | `fcntl` lock; merges are fast-forward only; on a conflict the branch is kept, with no LLM repair |
| `kev_memory/compact_handoff.py` | idea from `src/agent/reflection-runs.ts` | extractive, with no model calls |

**The context repository.**
- **Where:** `~/.kev-memory/<domain>.context/`, a git repo on branch `main`.
- **Layout:**
  - root `MEMORY.md` is the map;
  - `project.md` is the core file;
  - `notes/` holds handoff notes, with archived ones in `notes/archive/`;
  - `decisions/` holds the user's doc/code conflict decisions;
  - `.memfs.config.json` sets the limits.
- **Rules:** every write is one commit by `defrost-ai`. The pre-commit hook embeds the validator source, so it runs
  without the package installed.
- **Post-commit hook:** re-indexes the search domain `<domain>-context` and, if `kev-memory context remote URL` was
  set, pushes to the team remote.
- **Migration:** the old notes folder and the `conflicts.jsonl` log are migrated on the first write.

**Commands.**

| command | what it does |
|---|---|
| `kev-memory context init` | creates the repo and registers `<domain>-context` |
| `kev-memory context check` | runs the hook's validation on the working tree |
| `kev-memory context log` | the audit trail, also the MCP tool `memory_context_log` |
| `kev-memory context brief` | the root map plus core files, within a word budget |
| `kev-memory context defrag` | archives old notes, splits oversize files and rebuilds indexes, as a worktree job that fast-forwards `main` |
| `kev-memory context branches [--repo .]`, `context merge <branch>` | review and merge job branches |
| `kev-memory context remote URL\|none` | sets or removes the team remote |

**Changed behaviour.**
- **`--docs-auto`:** the background `claude -p /document-changes` now runs in a worktree of the project on
  `defrost/docs/<sha>`. The branch waits for review by default; `--docs-auto-merge` fast-forwards it when clean.
- **`--handoff-on-compact`:** a PreCompact hook writes an extractive handoff from the transcript: the first and
  latest request, the TodoWrite state, edited files and open questions.

**Writing rules.** Generated notes and decisions follow `docs/WRITING_FOR_EXTRACTION.md` for their bodies:
- one topic per section, with the goal named in each heading;
- names and paths in backticks;
- a `Facts:` block of `Subject → relation → Object` lines.

They use Letta's `name`/`description` frontmatter instead of the doc-page `type/entity/status/updated` fields, and
the hook enforces it. A test lints them with `doc_lint.py`.

**Not built.**
- **An LLM reflection pass** (Letta's sleep-time agent). It would plug into `context_repo.defrag` as an extra job
  step. It would be opt-in and budget-capped.
- **LLM conflict repair** for kept branches.
