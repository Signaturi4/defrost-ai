---
description: Search the project memory (docs + linked code) and answer with citations
argument-hint: <question>
allowed-tools: Bash(kev-memory search:*), mcp__defrost__memory_search, mcp__defrost__memory_resolve_conflict, Read, AskUserQuestion
---
Search the project memory for: $ARGUMENTS

1. Call the `memory_search` MCP tool with the question (k "auto"). If the tool is not available, run
   `kev-memory search "$ARGUMENTS" -k auto`.
2. Read the returned sections. They are documentation, and documentation can be out of date. The first line of the
   result gives this project's **doc trust**: HIGH means answer from the sections and open code only for `!` lines;
   LOW means the sections are hints and step 3 always applies.
3. Ground the answer in the code (doc trust LOW: always; HIGH: only for `!` lines). When the question is about behaviour, configuration, deploy, CI, schedules,
   ports, limits or defaults, open the files on the `verify in:` lines with Read (usually 1–3, config files first)
   before answering. Always open the files behind a `!` line: `doc may be stale` (the file changed after the doc)
   or `doc/code conflict` (the doc names files or functions the code no longer has).
4. Answer. Cite docs as `path:Lstart-end` and code as `path:line`. When the doc and the code disagree, do not
   decide which is right: state both versions, then ask the user with AskUserQuestion, one question per conflict
   (header "Conflict"): "Code is right: update the doc", "Doc is right: the code is a bug", "Not a conflict",
   "Not sure: mark as open question". Record each answer with `memory_resolve_conflict`, then act on it: update the
   doc section; or report the bug (change code only if the user asks); or nothing; or add an open-question note.
   Hits with a `resolved:` line were decided before: follow that and do not ask again.
5. If nothing relevant comes back, say so and suggest `/memory-update` or a `bm25` search with an exact identifier.
