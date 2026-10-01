---
description: Refresh the project memory after docs or code changed (incremental)
argument-hint: [domain]
allowed-tools: Bash(defrost update:*), Bash(defrost domains:*), mcp__defrost__memory_update, mcp__defrost__memory_domains
---
Refresh the memory for domain "$ARGUMENTS" (if empty, use the domain whose source path contains the current
directory; list them with `memory_domains`).

Call the `memory_update` MCP tool (or run `defrost update <domain>`) and report the counts and how many sections
changed.
