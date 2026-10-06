# Changelog

## Unreleased

- **Fix: the search service could run code from the directory it was started in.** It started with
  `python -m defrost_ai.cli serve` in the caller's working directory, and `python -m` imports from there first: a
  service started inside a checkout with its own `defrost_ai/` folder ran that code, and later starts kept using it
  as "another installation". The service now starts from its installation's root.
- **Fix: the search service grew without bound on Apple Silicon** and could exhaust memory and swap (a 24 GB Mac
  panicked with three services at 17-22 GB each). MLX kept every freed GPU buffer for reuse, and reranking makes new
  batch shapes on each query: about 0.5 GB more per reranked search. The MLX buffer cache is now capped at 512 MB
  (`DEFROST_MLX_CACHE_MB`); latency is unchanged (0.35-0.4 s per reranked query at 0, 512 MB and 2 GB).
- **The prompt hook finds both sides of a conflict:** it now searches reranked (`prompt_context.mode accurate`,
  falling back to fast when the reranker is cold), adds the top 5 sections (`prompt_context.k`) within 3,500 tokens
  (`prompt_context.budget_tokens`), and tells Claude to report every disagreement with both versions and to avoid a
  plain yes or no the evidence only partly supports. Low doc trust now states both versions before asking. On a
  real repository (20 questions × 2 runs, blind judge): fully correct 100% (fast top 3: 94% on the 8 hardest), conflict
  traps flagged 100% (was 67%), hallucination 0%; cost per run about +10%. Details: `docs/AGENT_QA.md`.
- **Agent QA benchmark** (`bench/agent_qa`): Claude Code headless per arm, a blind judge that scores against key
  points, hook-output capture, and a paired report.
- **Fix: a settings section split in two made `config.toml` unreadable** (duplicate TOML table, every setting fell
  back to its default); settings of a section now stay together, with a test.
- **`defrost forget NAME` deletes a memory:** its built index and previous build, workspace file, trigger
  state, logs, schedule and registry entry, plus the hooks it installed in its repository unless another memory
  indexes the same repository (worktrees share git hooks). Only files inside the defrost home are deleted; project
  files and the notes repository stay. `--dry-run` lists what would go. Before, a stray memory could only be
  removed by hand.
- **The skill lives in its own repository:** `skills/knowledge_lifecycle_skill/` is a git submodule of
  [knowledge_lifecycle_skill](https://github.com/Signaturi4/knowledge_lifecycle_skill), so defrost carries exactly
  the published skill, tests and evals; the copies under `skills/extraction-ready-docs/` and `skills/tests/` are gone.
- **Doc rules as a standalone skill:** `extraction-ready-docs/` packages the doc-writing rules, templates,
  linter and Facts extractor as an agent skill that works without defrost. Its `scripts/install.py` is Python-only
  (macOS, Linux, Windows), can also write `AGENTS.md`, and removes the block with `--remove`. Its linter accepts
  directories and skips `templates/` and `DOC_RULES.md`.
- **Knowledge-management rules in the skill:** `references/KNOWLEDGE_RULES.md` adds one home per file (docs map and
  folder indexes), what-who-when file names with full `YYYY-MM-DD` dates (optional `docs/CODES.md` prefixes for 100+
  files), a lifecycle per file (`living`, `versioned` with `archive/`, `immutable`; default: the agent decides per
  page), `source: true` files with a Sources section, text twins for binaries, a decision register, folder splits
  and safe moves. New `scripts/repo_lint.py` checks names, duplicate copies and entities, lifecycle, indexes, twins
  and references. `install.py` is one command for any repository (it creates the `docs/README.md` map and
  updates `AGENTS.md` when present), plus `--defrost` (linters, lint rule and defrost markers) and an optional
  `--lifecycle`. After installing it scans every `.md` file (`scripts/audit.py`) and splits the work into in-place
  fixes, applied right away, and renames/moves/merges, proposed for a yes. Tests and behaviour evals live in the
  skill repository's `tests/`, outside the package.

## 1.2.1 (2026-10-02)

- **Questions are answered from the memory in one turn:** a Claude Code `UserPromptSubmit` hook
  (`defrost hook prompt`, on in the `standard` and `full` profiles, `--no-prompt-context` to turn it off) searches
  the memory for each prompt (fast mode, top 3) and adds the sections before Claude starts. Measured on a docs
  repository for "who is artem": 18-30 s and 5 model turns before, ~4 s and 1 turn with Sonnet. It skips slash
  commands, prompts under 3 words and prompts whose best section is below `prompt_context.min_cosine` (0.34;
  calibrated on 10 real questions, 0.35-0.60, against 13 coding and chit-chat prompts, 0.00-0.32), and never waits
  for a cold service.
- **Doc trust is a required setup question:** it is asked for every new project instead of the default search mode
  (still `--mode` / `defrost config search.mode`), and `--yes` without `--doc-trust` stops with the suggestion for
  the repository instead of silently choosing `low`. `defrost setup --suggest-trust` prints it: `high` for a docs
  repository (at least 10 doc files per code file), else `low`. The `/defrost-setup` command always asks it.
- **Fix: re-running setup no longer resets doc trust:** a re-run without `--doc-trust` passed the global default
  (`low`) and overwrote a project set to `high`; the stored level is now kept.

## 1.2.0 (2026-10-02)

- **`mcp` is a core dependency:** the MCP server installs with defrost-ai; the `[mcp]` extra is kept, empty, so
  existing install commands still resolve. If an environment still lacks it, `defrost mcp` exits with the reinstall
  command instead of a traceback that MCP clients only show as "connection closed"; `defrost status` and Claude
  registration warn too.
- **Portable Claude hooks:** `.claude/settings.json` calls `defrost` from `PATH` (no machine-specific path) and skips
  silently where defrost is not installed, so the file can be committed for teammates.
- **A stale memory no longer breaks search:** a registered memory whose workspace file was deleted is listed as not
  built (with the reason) instead of failing `defrost search` across all memories with HTTP 500.
- **Fast search across memories ranks by relevance:** each memory's #1 hit used to tie, so the first-registered
  memories filled the results whatever they matched; hits are now merged by cosine, keeping each memory's own order.
- **The MCP server says when it was replaced:** a running `defrost mcp` whose install changed under it (reinstall or
  upgrade, e.g. under another Python) used to fail each tool call with "cannot import name ..."; it now answers
  "reconnect it: /mcp → defrost → Reconnect". MCP clients own the process, so it cannot restart itself.
- **Upgrades restart the service after a Python change:** a running service whose install directory no longer exists
  (reinstalled under another Python) is restarted instead of being kept as "another installation" running old code.
- **Rename:** the package is `defrost-ai` (import `defrost_ai`), the CLI is `defrost`, and the models are
  Defrost-Ret-B and Defrost-Rerank v2. `kev-memory`, `KEV_*` variables and `~/.kev-memory` keep working until 1.3.
- **Simpler surface:**
  - 4 MCP tools: `search`, `docs_for`, `remember`, `refresh`.
  - 9 commands; `defrost setup` asks 3 questions, or takes `--yes`.
  - 2 search modes: `accurate` and `fast`. The 1.1 `fast` policy is now called `accurate`.
  - One settings file, `~/.defrost-ai/config.toml`.
- **Doc trust, conflicts and working memory:**
  - Doc trust levels (`low` / `high`).
  - Doc/code conflicts are decided by the user and recorded.
  - Working memory lives in `<project>/defrost-memory/`: notes and decisions, with Letta-style layout rules.
- **Installer:** pinned to a release tag (`DEFROST_VERSION`, `DEFROST_REF`, `DEFROST_REPO`), clear errors, and
  `defrost --version`. A missing weights release gives a clear message instead of a traceback.
- **Docs:**
  - The leakage statement is corrected: private-repo scores are in-domain.
  - Release checklist with a data-provenance gate (`docs/RELEASING.md`, `scripts/release_check.py`).
- Weights: unchanged, still v1.1.0.

## 1.1.0

- Defrost-Rerank v2 (then Kev-Rerank v2) as the default reranker. Weights archive v1.1.0.
- **Speed:**
  - Merged-weights cache.
  - Length-sorted batches.
  - MLX fp16 backend on Apple Silicon.
  - Score cache.
  - CLI search through the resident service.
  - A warm reranked search went from 7.7 s to about 1.7 s on an M5, with fp32 quality kept.
- One-line install, setup triggers (refresh on merge, schedule, session start), MCP server and slash commands.
- Grounding: config/CI file links, `verify in:` lines, stale-doc and conflict flags, and more precise doc→code links.

## 1.0.0

- First public release: hybrid BM25 + dense + rerank search over doc sections with doc→code links, as a CLI,
  service and Python API. Weights archive v1.0.0.
