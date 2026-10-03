# Monitoring mode

A log of how Claude Code uses the project memory in each session, for debugging and later analysis. Optional,
off by default, local only.

## Turn it on and off

1. Install the hooks once per clone: `defrost setup . --monitor` (with your usual setup flags). They do nothing until
   step 2. `--no-monitor` removes them; `defrost setup --remove` removes them with every other trigger.
2. Switch it in the project's `.env`, which is not committed:

   ```sh
   DEFROST_MONITOR=on     # on | off   (also 1/0, true/false, yes/no)
   ```

   The environment wins over `.env`, so `DEFROST_MONITOR=off claude` turns it off for one session.

Off costs nothing: the hook command checks the switch in the shell and starts no Python. On, each event adds one
short `defrost hook monitor` run of about 0.1 s.

**Safe by construction**
- The hook always exits 0 and prints nothing. It can never block a tool call or change what Claude sees.
- Deleting `defrost_ai/monitor.py`, or uninstalling defrost, leaves everything else working. The hooks become
  no-ops, and defrost's own code reaches the monitor only through `defrost_ai.monitor_event`, which ignores a
  missing module. The test suite passes without the file.

## What it logs

One JSON object per line, one file per Claude session: `<DEFROST_HOME>/monitor/<project>/<session>.jsonl`.

| event | when | fields |
|---|---|---|
| `session_start` / `session_end` | session opens / closes | source, model, cwd, git branch; end: reason + summary |
| `prompt` | you send a message | text (preview), chars, words |
| `memory_inject` | the prompt hook ran | injected / skipped / error, why, best cosine, hits, ms |
| `tool_start` / `tool_end` | every tool call | tool, category, memory command, input / output preview, duration_ms, error |
| `trace` | at Stop, from Claude's transcript | the full sequence: `user_input`, `reasoning`, `assistant_text`, `tool_call`, `tool_result` (with token usage per message, subagent steps marked) |
| `stop`, `subagent_stop`, `pre_compact`, `notification` | Claude Code events | trigger, open tool calls |
| `defrost_hook` | every other defrost hook | hook, ms, ok / error |
| `monitor_error` | the monitor itself failed | error |

Events without a Claude session go to `_service-<date>.jsonl` in the same folder, with pid and cwd. Today that is
MCP tool calls (`mcp_tool`: tool, ms, ok / error, arguments, result size).

Tool categories: `grep` (Grep, Glob, and Bash commands that run grep, rg, ag, ack, git grep, find or fd),
`memory_search` (MCP `search` / `docs_for`, `defrost search|docs|status`), `memory_update` (MCP `refresh` /
`remember`, `defrost refresh|note|setup`), `read`, `edit`, `shell`, `web`, `agent`, `mcp`, `other`.

## Read it

```sh
defrost monitor status                       # switch, hooks, log folder, number of sessions
defrost monitor report                       # the last session: counts, memory use, errors, timings, tokens
defrost monitor report --last 5 --sequence   # plus the step-by-step sequence of each session
defrost monitor report --json                # for scripts and notebooks
```

The report counts:
- grep calls; memory calls by command (`mcp:search x3, cli:defrost refresh x1`); memory updates;
- prompts that got memory injected, and why the others did not;
- tool errors and tool calls that never finished;
- median and max duration per category;
- compactions, subagents, and tokens.

`--sequence` prints the steps in order:

```text
USER      who is artem
  MEMORY-INJECT injected  best=0.42 270ms
  REASON  (412 chars) The memory sections answer it ...
  TOOL    Bash [grep] {"command": "grep -rn artem ."}
  RESULT ERROR (0 chars)
  SAY     Artem Zelenskii is ...
```

## Settings

Set these in the environment or in `.env`.

| key | default | meaning |
|---|---|---|
| `DEFROST_MONITOR` | off | the switch |
| `DEFROST_MONITOR_DIR` | `<DEFROST_HOME>/monitor` | log folder |
| `DEFROST_MONITOR_MAX_CHARS` | 2000 | preview length for prompts, tool input and output, reasoning; `0` = counts and timings only |

**Privacy.** The logs hold prompt text, tool output and Claude's reasoning (previews, cut at
`DEFROST_MONITOR_MAX_CHARS`). They stay on this machine, outside the repository. Set `DEFROST_MONITOR_MAX_CHARS=0`
to keep only counts and timings.

## Platforms

- **macOS:** tested.
- **Windows:** reviewed, not executed. The hook command is POSIX shell (`command -v`, `case`, `grep -E`), which
  Claude Code runs through Git Bash on Windows. Without Git Bash it does nothing, and Claude keeps working.
