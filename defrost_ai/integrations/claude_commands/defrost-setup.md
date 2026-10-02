---
description: Set up defrost-ai for this repository in under a minute (3 questions, then it builds the memory)
allowed-tools: Bash(defrost setup:*), Bash(defrost status:*), Bash(pwd), AskUserQuestion
---
Set up defrost-ai for the current repository. Keep it quick and friendly: one round of questions, then build.

1. Run `pwd`, `defrost status` and `defrost setup . --suggest-trust` (prints the recommended doc trust, why, and the
   level stored for this repository, if any). If this repository already has a memory, say so in one line (sections,
   when it was built, which refresh is installed, its doc trust) and ask only whether to change the setup; doc trust
   is still asked when the stored level differs from the suggested one.
2. Ask these 3 questions in ONE AskUserQuestion call. Put the recommended option first:
   - **"How much should defrost do here?"** (header "Setup")
     - "Standard (Recommended)": keeps the memory fresh after every merge or commit to main, adds short
       doc-writing rules to CLAUDE.md, starts a new session from your handoff note after /clear, and reminds you of
       commits whose docs need an update.
     - "Minimal": only keeps the memory fresh after merges and commits to main.
     - "Full": Standard, and Claude also updates the affected docs before it finishes and before it commits, and a
       handoff note is written automatically before compaction.
   - **"How much should Claude trust this project's docs?"** (header "Doc trust"). MANDATORY: always ask it, never
     pick it for the user, never pass `--yes` without `--doc-trust`. Put the `--suggest-trust` level first with
     "(Recommended)" and its reason in the description:
     - "Docs are reliable": Claude answers from the docs and checks code only when a doc is flagged as stale. Right
       for docs repositories and well-maintained docs.
     - "Code is the truth": docs are hints; Claude checks the linked code before answering. Right for fast-moving
       code with lagging docs.
   - **"Search the memory automatically for each question?"** (header "Speed"). Skip it for "Minimal".
     - "Yes (Recommended)": before Claude answers, the memory's best sections are added to the question, so a lookup
       is answered in one turn (~4 s instead of ~20 s). Commands, short replies and unrelated prompts are skipped.
     - "No": Claude calls the search tool itself when it decides to.
   The search mode is not asked: it defaults to accurate, and `defrost config search.mode fast` changes it.
3. Run once, with the Bash timeout set to 600000 ms (the first build loads the models; tell the user it is running):
   `defrost setup . --yes --profile standard|minimal|full --doc-trust low|high [--no-prompt-context]`
   If it times out, rerun with `--build background` and check `defrost status` later.
4. Reply in 4–6 short lines: what was built (sections, doc→code links), what refreshes it, the doc trust, whether
   questions are searched automatically, and how to use it: "just ask how something works; I search the memory
   first", `/ask <question>` to force a search, `/handoff` before /clear. Changing later: `defrost setup --profile …`, `defrost config`,
   `defrost setup --remove`.
