---
description: Save a handoff note to the project memory so you can /clear and continue with a small context
allowed-tools: mcp__defrost__remember, Bash(defrost note:*)
---
Write a handoff note for the current work with `remember(kind="note", ...)` (no tool: `defrost note "goal" --state ... --next ... --why ... --file ...`), then tell the user they can run `/clear`.
$ARGUMENTS

Fill it from this conversation, briefly and concretely. The next session sees only this note, CLAUDE.md and the
project memory, not the conversation.
- **goal**: the user's goal in one paragraph, with acceptance criteria and any constraints the user stated
  ("do not push", "ask before spending tokens").
- **state**: what is done, what is verified and how (tests, commands, numbers), and what is in progress.
- **decisions**: each with its reason, e.g. "bf16 only for the reranker: the query encoder must match the stored
  fp32 vectors".
- **next_steps**: concrete actions in order; name the commands.
- **files**: paths that were changed or must be read first.

Leave out tool output, file contents and dead ends unless a dead end prevents a repeated mistake. Stay under about
300 words in total. Then reply with one line: the note's path and "run /clear to continue from it".
