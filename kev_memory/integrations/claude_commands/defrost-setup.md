---
description: Set up defrost-ai project memory for this repo (asks how it should stay fresh, then builds it)
allowed-tools: Bash(kev-memory setup:*), Bash(kev-memory status:*), Bash(git rev-parse:*), Bash(git log:*), Bash(pwd), AskUserQuestion
---
Set up the defrost-ai project memory for the current repository. Do every step below; ask the user only through
the AskUserQuestion tool (it takes at most 4 questions per call): the four main questions in ONE call, and the
interval question in a second call only when "Every few hours" was picked.

1. Run `pwd`, `git rev-parse --abbrev-ref HEAD` and `kev-memory status`. If this path already has a domain with
   triggers, show its status and ask only whether to change the triggers (re-running setup replaces them).
2. Ask these questions (AskUserQuestion; "Refresh", "Doc trust", "Doc sync", "Doc rules" in one call, then
   "Interval" alone if needed):
   - **"When should the memory refresh?"** (multiSelect, header "Refresh")
     - "Build now + on every merge/commit to main (Recommended)": git hooks; runs only when a doc or code file changed.
     - "Every few hours": a background schedule (launchd on macOS, cron on Linux).
     - "When a Claude session starts": a SessionStart hook; refreshes in the background only if stale.
     - "Only when I run /memory-update": no automatic trigger.
   - **"How often, if on a schedule?"** (header "Interval"): "Every 6 hours (Recommended)", "Every 2 hours",
     "Every 24 hours".
   - **"How much should Claude trust this project's docs?"** (header "Doc trust"):
     "Code is the truth (Recommended for startups)": docs are hints; Claude always reads the linked code before
     answering (best when code changes daily and docs lag; costs a few file reads per question);
     "Docs are reliable (legacy, well documented)": Claude answers from the docs and reads code only when the memory
     flags a doc as stale or conflicting (fewer tokens; only safe when docs are kept up to date).
     Mark "Code is the truth" as recommended unless the repo's docs are clearly maintained (e.g. most doc files were
     changed in the last month, per `git log`); then recommend "Docs are reliable".
   - **"How should Claude keep the docs in step with the code?"** (multiSelect, header "Doc sync")
     - "Document changes after edits and before each commit (Recommended)": project hooks; when code changed,
       Claude is asked once per change (on finishing, and before `git commit`) to update the doc sections that
       describe it and to document new commands, env vars and files. Commits made outside Claude are recorded as
       pending doc tasks and shown at the next session start. No extra model calls.
     - "Also write the docs automatically after commits made outside Claude": runs `claude -p /document-changes`
       in the background after each such commit, in a separate git worktree on a review branch
       `defrost/docs/<sha>` (your checkout is never touched; merge it with `kev-memory context merge <branch>
       --repo .`); capped at $0.50 per commit. Costs Claude usage.
     - "Start from a handoff note after /clear": `/handoff` saves goal, state and next steps as a commit in the
       project's context repository (a small git repo of notes and decisions); after `/clear` the new session
       starts from that note instead of the old conversation.
     - "Also save a handoff note automatically before each compaction": no model calls; the goal, todo list,
       edited files and open questions are taken from the transcript.
   - **"Add the doc-writing rules to CLAUDE.md?"** (header "Doc rules"): "Yes (Recommended)": a short highlighted
     block at the end of CLAUDE.md, with full rules in docs/DOC_RULES.md that load only when docs are written;
     "No".
3. Build the command from the answers and run it once:
   `kev-memory setup . --build now --doc-trust low|high [--on-main-merge] [--every-hours N] [--claude-hook] [--doc-rules]
   [--docs-sync] [--docs-auto] [--handoff]`
   ("Build now + main" → `--on-main-merge`; "Every few hours" → `--every-hours N` from the interval answer;
   "session starts" → `--claude-hook`; doc rules "Yes" → `--doc-rules`; "Code is the truth" → `--doc-trust low`,
   "Docs are reliable" → `--doc-trust high`; "Document changes…" → `--docs-sync`;
   "Also write the docs automatically…" → `--docs-auto`; "handoff note" → `--handoff`; "before each compaction" →
   `--handoff-on-compact`). Doc sync needs the doc
   rules: if it is chosen, add `--doc-rules` too. Run it with the Bash timeout set to
   600000 ms: the first build loads the models and can take several minutes on a large repo; tell the user it is
   running. If it still times out, rerun with `--build background` and check `kev-memory status` later.
4. Report: sections, code nodes and doc→code links from the build, the triggers installed, and the commands to use:
   `/memory-search <question>`, `/memory-update`, `/memory-domains`, `/document-changes [sha]`, `/handoff`; `kev-memory status` shows staleness;
   `kev-memory setup . --remove-triggers` removes all triggers.
