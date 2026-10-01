---
description: Search the project memory (docs + linked code) and answer with citations
argument-hint: <question>
allowed-tools: Bash(kev-memory search:*), mcp__defrost__memory_search, Read
---
Search the project memory for: $ARGUMENTS

1. Call the `memory_search` MCP tool with the question (k "auto"). If the tool is not available, run
   `kev-memory search "$ARGUMENTS" -k auto`.
2. Answer from the returned sections only. Cite each claim as `path:Lstart-end`.
3. Name the code the sections link to (the `-> code` lines). Open a file with Read only if the sections do not
   answer the question.
4. If nothing relevant comes back, say so and suggest `/memory-update` or a `bm25` search with an exact identifier.
