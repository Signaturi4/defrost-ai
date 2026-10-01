---
description: Set up defrost-ai for this repository in under a minute (3 questions, then it builds the memory)
allowed-tools: Bash(defrost setup:*), Bash(defrost status:*), Bash(git log:*), Bash(pwd), AskUserQuestion
---
Set up defrost-ai for the current repository. Keep it quick and friendly: one round of questions, then build.

1. Run `pwd` and `defrost status`. If this repository already has a memory, say so in one line (sections, when it
   was built, which refresh is installed) and ask only whether to change the setup.
2. Ask these 3 questions in ONE AskUserQuestion call. Put the recommended option first:
   - **"How much should defrost do here?"** (header "Setup")
     - "Standard (Recommended)": keeps the memory fresh after every merge or commit to main, adds short
       doc-writing rules to CLAUDE.md, starts a new session from your handoff note after /clear, and reminds you of
       commits whose docs need an update.
     - "Minimal": only keeps the memory fresh after merges and commits to main.
     - "Full": Standard, and Claude also updates the affected docs before it finishes and before it commits, and a
       handoff note is written automatically before compaction.
   - **"Default search mode?"** (header "Search")
     - "Accurate (Recommended)": best results. A reranker double-checks when the two retrievers disagree,
       about 1–2 s per search on Apple Silicon.
     - "Fast": about 0.1 s per search, without the reranker; a little less accurate.
   - **"How much should Claude trust this project's docs?"** (header "Doc trust")
     - "Code is the truth (Recommended for fast-moving projects)": docs are hints; Claude checks the linked code
       before answering.
     - "Docs are reliable": Claude answers from the docs and checks code only when a doc is flagged as stale.
     Recommend "Docs are reliable" first instead only if `git log --since=30.days -- '*.md'` shows the docs are
     actively maintained.
3. Run once, with the Bash timeout set to 600000 ms (the first build loads the models; tell the user it is running):
   `defrost setup . --yes --profile standard|minimal|full --mode accurate|fast --doc-trust low|high`
   If it times out, rerun with `--build background` and check `defrost status` later.
4. Reply in 4–6 short lines: what was built (sections, doc→code links), what refreshes it, the search mode and doc
   trust, and how to use it: "just ask how something works; I search the memory first", `/ask <question>` to force
   a search, `/handoff` before /clear. Changing later: `defrost setup --profile …`, `defrost config`,
   `defrost setup --remove`.
