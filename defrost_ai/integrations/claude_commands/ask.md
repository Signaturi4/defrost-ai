---
description: Answer a question from the project memory (docs + linked code), with citations
argument-hint: <question> [--fast]
allowed-tools: Bash(defrost search:*), mcp__defrost__search, mcp__defrost__remember, Read, AskUserQuestion
---
Answer: $ARGUMENTS

1. Call the `search` tool with the question (mode "fast" only if the user wrote --fast). Without the tool, run
   `defrost search "<question>"`.
2. The first line of the result is this project's **doc trust**. LOW: the sections are hints; open the files on the
   `verify in:` lines (1–3, config files first) and answer from the code. HIGH: answer from the sections; open code
   only for hits with a `!` line. Always open the files behind a `!` line.
3. Answer briefly. Cite docs as `path:Lstart-end` and code as `path:line`.
4. If a doc and the code disagree (verified in the code; a `! doc may be stale` line alone is not a question), do not
   pick a side. Show both, then ask with AskUserQuestion (header "Conflict"), at most 2 per answer; list the rest at the end:
   "Code is right: update the doc", "Doc is right: the code is a bug", "Not a conflict", "Not sure: mark as open".
   Save the answer with `remember(kind="decision", ...)` and act on it (change code only if the user asks).
   Hits with a `resolved:` line were already decided: follow them.
5. If nothing relevant comes back, say so, and suggest the `refresh` tool or a more specific question.
