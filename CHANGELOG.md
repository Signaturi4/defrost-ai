# Changelog

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
