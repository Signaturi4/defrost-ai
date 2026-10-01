---
description: Set up defrost-ai project memory for this repo (asks how it should stay fresh, then builds it)
allowed-tools: Bash(kev-memory setup:*), Bash(kev-memory status:*), Bash(git rev-parse:*), Bash(pwd), AskUserQuestion
---
Set up the defrost-ai project memory for the current repository. Do every step below; ask the user only through
the AskUserQuestion tool, all questions in ONE call.

1. Run `pwd`, `git rev-parse --abbrev-ref HEAD` and `kev-memory status`. If this path already has a domain with
   triggers, show its status and ask only whether to change the triggers (re-running setup replaces them).
2. Ask these questions (AskUserQuestion, one call):
   - **"When should the memory refresh?"** (multiSelect, header "Refresh")
     - "Build now + on every merge/commit to main (Recommended)": git hooks; runs only when a doc or code file changed.
     - "Every few hours": a background schedule (launchd on macOS, cron on Linux).
     - "When a Claude session starts": a SessionStart hook; refreshes in the background only if stale.
     - "Only when I run /memory-update": no automatic trigger.
   - **"How often, if on a schedule?"** (header "Interval"): "Every 6 hours (Recommended)", "Every 2 hours",
     "Every 24 hours".
   - **"How should Claude keep the docs in step with the code?"** (multiSelect, header "Doc sync")
     - "Document changes after edits and before each commit (Recommended)": project hooks; when code changed,
       Claude is asked once per change (on finishing, and before `git commit`) to update the doc sections that
       describe it and to document new commands, env vars and files. Commits made outside Claude are recorded as
       pending doc tasks and shown at the next session start. No extra model calls.
     - "Also write the docs automatically after commits made outside Claude": runs `claude -p /document-changes`
       in the background after each such commit; it edits docs only, never commits, and is capped at $0.50 per
       commit. Costs Claude usage.
     - "Start from a handoff note after /clear": `/handoff` saves goal, state and next steps; after `/clear` the
       new session starts from that note instead of the old conversation.
   - **"Add the doc-writing rules to CLAUDE.md?"** (header "Doc rules"): "Yes (Recommended)": a short highlighted
     block at the end of CLAUDE.md, with full rules in docs/DOC_RULES.md that load only when docs are written;
     "No".
3. Build the command from the answers and run it once:
   `kev-memory setup . --build now [--on-main-merge] [--every-hours N] [--claude-hook] [--doc-rules] [--docs-sync]
   [--docs-auto] [--handoff]`
   ("Build now + main" → `--on-main-merge`; "Every few hours" → `--every-hours N` from the interval answer;
   "session starts" → `--claude-hook`; doc rules "Yes" → `--doc-rules`; "Document changes…" → `--docs-sync`;
   "Also write the docs automatically…" → `--docs-auto`; "handoff note" → `--handoff`). Doc sync needs the doc
   rules: if it is chosen, add `--doc-rules` too. Run it with the Bash timeout set to
   600000 ms: the first build loads the models and can take several minutes on a large repo; tell the user it is
   running. If it still times out, rerun with `--build background` and check `kev-memory status` later.
4. Report: sections, code nodes and doc→code links from the build, the triggers installed, and the commands to use:
   `/memory-search <question>`, `/memory-update`, `/memory-domains`, `/document-changes [sha]`, `/handoff`; `kev-memory status` shows staleness;
   `kev-memory setup . --remove-triggers` removes all triggers.
