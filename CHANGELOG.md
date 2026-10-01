# Changelog

## 1.2.0 (unreleased)

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
