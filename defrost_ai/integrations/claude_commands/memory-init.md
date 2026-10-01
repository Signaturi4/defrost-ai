---
description: Create the project memory for this repo (docs + code), stored outside the repo
argument-hint: [domain-name]
allowed-tools: Bash(defrost build:*), Bash(pwd), mcp__defrost__memory_init
---
Create a memory for the current repository. Domain name: "$ARGUMENTS" (if empty, use the repo folder name in
lower case).

Call the `memory_init` MCP tool with the absolute path of the current directory and that domain. Report the
section, code-node and doc→code link counts when it finishes. Then suggest adding the doc rules
(`templates/doc-rules/install.sh`) so new docs are written in a form the memory reads well.
