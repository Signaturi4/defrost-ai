"""Monitoring mode: a JSONL log of how Claude Code uses the project memory, for debugging and later analysis.

Optional and self-contained. Off unless DEFROST_MONITOR is on (environment, or the project's `.env`):

    DEFROST_MONITOR=on              # .env in the project root (or exported); off / 0 / false / no turns it off

`defrost setup . --monitor` adds one hook entry per Claude Code event to .claude/settings.json. The hook command
checks the switch in the shell first, so a disabled monitor costs no Python start; it always exits 0 and prints
nothing, so it can never block a tool call or change what Claude sees. Deleting this file (or uninstalling defrost)
leaves everything else working: the hooks become no-ops and defrost's own code reaches this module only through
`defrost_ai.monitor_event`, which ignores a missing module.

What is logged, one JSON object per line, per Claude session in <DEFROST_HOME>/monitor/<project>/<session>.jsonl:
    session_start / session_end   source, model, cwd, git branch; end: reason and a summary
    prompt                        the user's input (truncated)
    tool_start / tool_end         every tool call: name, category (grep, memory_search, memory_update, read, edit,
                                  shell, web, agent, mcp, other), the memory command, input / output previews,
                                  duration, error flag
    trace                         the full execution sequence read from Claude's transcript at Stop: user input,
                                  reasoning (thinking + text), tool calls, tool results, token usage per message
    stop / subagent_stop / pre_compact / notification
    memory_inject                 the prompt hook's decision (injected or why not), best cosine, latency
    defrost_hook / mcp_tool       every defrost hook and MCP tool call: duration, ok / error
Events without a Claude session (the MCP server) go to <DEFROST_HOME>/monitor/<project>/_service-<date>.jsonl.

    defrost monitor report [--last N | --session ID] [--sequence]    counts, reliability, the execution sequence
    defrost monitor status                                           switch, hooks, log folder

Settings (environment or .env): DEFROST_MONITOR, DEFROST_MONITOR_DIR (log folder), DEFROST_MONITOR_MAX_CHARS
(preview length, default 2000; 0 = no previews, counts and timings only). The logs contain prompts and tool output
previews: they stay on this machine and are never committed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HOOK_TAG = "defrost-ai:monitor"
EVENTS = {"SessionStart": None, "SessionEnd": None, "UserPromptSubmit": None, "PreToolUse": "*", "PostToolUse": "*",
          "Stop": None, "SubagentStop": None, "PreCompact": None, "Notification": None}
ON = {"1", "true", "on", "yes"}
OFF = {"0", "false", "off", "no"}

# ---- switch -------------------------------------------------------------------------------------------------------
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*?)\s*$")


def _project_root(path: str | Path | None) -> Path:
    p = Path(path or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).expanduser().resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists() or (d / ".env").exists():
            return d
    return p


def dotenv(root: str | Path | None) -> dict[str, str]:
    """KEY=value pairs of the project's .env (quotes stripped, comments ignored); {} when there is none."""
    out = {}
    try:
        lines = (_project_root(root) / ".env").read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if m and m.group(1).startswith("DEFROST_MONITOR"):
            out[m.group(1)] = m.group(2).split(" #")[0].strip().strip("'\"")
    return out


def setting(key: str, root=None, default: str = "") -> str:
    """The environment wins over the project's .env, so `DEFROST_MONITOR=off claude` turns it off for one run."""
    return os.environ.get(key) or dotenv(root).get(key) or default


def enabled(root=None) -> bool:
    return setting("DEFROST_MONITOR", root).strip().lower() in ON


def _max_chars(root=None) -> int:
    try:
        return max(0, int(setting("DEFROST_MONITOR_MAX_CHARS", root, "2000")))
    except ValueError:
        return 2000


# ---- log files ------------------------------------------------------------------------------------------------------
def log_dir(root=None) -> Path:
    base = setting("DEFROST_MONITOR_DIR", root)
    base = Path(base).expanduser() if base else \
        Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser() / "monitor"
    r = _project_root(root)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", r.name) + "-" + hashlib.sha1(str(r).encode()).hexdigest()[:6]
    return base / slug


def _now() -> tuple[float, str]:
    t = time.time()
    return t, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:                 # one line per write: safe for concurrent appends
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _session_file(session: str, root=None) -> Path:
    return log_dir(root) / f"{re.sub(r'[^A-Za-z0-9_-]', '_', session or 'no-session')}.jsonl"


def _preview(value, n: int) -> str | None:
    if n <= 0 or value is None:
        return None
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


def _size(value) -> int:
    if value is None:
        return 0
    return len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))


def emit(event: str, session: str | None = None, root=None, **fields) -> None:
    """Append one event; silent no-op when monitoring is off. Never raises."""
    try:
        if not enabled(root):
            return
        t, ts = _now()
        rec = {"ts": ts, "t": round(t, 3), "event": event, "session": session, **fields}
        if session:
            _write(_session_file(session, root), rec)
        else:
            _write(log_dir(root) / f"_service-{time.strftime('%Y-%m-%d')}.jsonl", rec | {"pid": os.getpid(),
                                                                                      "cwd": os.getcwd()})
    except Exception:                                            # noqa: BLE001  (monitoring must never break work)
        pass


# ---- classification ---------------------------------------------------------------------------------------------------
_GREP = re.compile(r"(^|[\s|;&(`])(grep|egrep|fgrep|rg|ag|ack|git\s+grep|find|fd)\s")
_DEFROST_CLI = re.compile(r"(^|[\s|;&(`/])(defrost|kev-memory)\s+([a-z-]+)")
_MEMORY_SEARCH = {"search", "docs", "docs-for", "status"}
_MEMORY_UPDATE = {"refresh", "note", "setup", "update", "build", "rollback"}


def classify(tool: str, tool_input: dict | None) -> tuple[str, str | None]:
    """-> (category, memory command or None). Bash commands are classified by what they run."""
    ti = tool_input or {}
    if tool.startswith("mcp__defrost__") or tool.startswith("mcp__kev-memory__"):
        name = tool.rsplit("__", 1)[-1]
        return ("memory_update" if name in ("refresh", "remember") else "memory_search"), f"mcp:{name}"
    if tool in ("Grep", "Glob"):
        return "grep", None
    if tool == "Bash":
        cmd = str(ti.get("command", ""))
        m = _DEFROST_CLI.search(cmd)
        if m:
            sub = m.group(3)
            if sub in _MEMORY_UPDATE:
                return "memory_update", f"cli:{m.group(2)} {sub}"
            if sub in _MEMORY_SEARCH:
                return "memory_search", f"cli:{m.group(2)} {sub}"
        if _GREP.search(" " + cmd):
            return "grep", None
        return "shell", None
    if tool == "Read":
        return "read", None
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        return "edit", None
    if tool in ("WebFetch", "WebSearch"):
        return "web", None
    if tool in ("Task", "Agent"):
        return "agent", None
    if tool.startswith("mcp__"):
        return "mcp", None
    return "other", None


def _is_error(response) -> bool:
    if isinstance(response, dict):
        if response.get("is_error") or response.get("error") or response.get("interrupted"):
            return True
        if isinstance(response.get("exit_code"), int) and response["exit_code"] != 0:
            return True
    return False


# ---- per-session state (tool start times, transcript offset) --------------------------------------------------------
def _state_path(session: str, root=None) -> Path:
    return log_dir(root) / ".state" / f"{re.sub(r'[^A-Za-z0-9_-]', '_', session or 'no-session')}.json"


def _load_state(session: str, root=None) -> dict:
    try:
        return json.loads(_state_path(session, root).read_text())
    except (OSError, ValueError):
        return {"starts": {}, "transcript_line": 0}


def _save_state(session: str, state: dict, root=None) -> None:
    p = _state_path(session, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state))


def _tool_key(payload: dict) -> str:
    return payload.get("tool_use_id") or hashlib.sha1(
        (payload.get("tool_name", "") + json.dumps(payload.get("tool_input"), sort_keys=True, default=str)).encode()
    ).hexdigest()[:16]


# ---- transcript: the full execution sequence ------------------------------------------------------------------------
def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def read_transcript(path: str | Path, start_line: int, n: int) -> tuple[list[dict], int]:
    """Transcript lines from start_line on -> trace steps in order: user_input, reasoning, assistant_text, tool_call,
    tool_result (with token usage on assistant steps) and the next line number to read."""
    steps, i = [], start_line
    try:
        lines = Path(path).expanduser().read_text(errors="replace").splitlines()
    except OSError:
        return steps, start_line
    for i, line in enumerate(lines[start_line:], start=start_line + 1):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        msg = e.get("message") or {}
        role, ts, side = e.get("type"), e.get("timestamp"), bool(e.get("isSidechain"))
        base = {"ts_model": ts, "sidechain": side}
        if role == "user":
            for b in _blocks(msg.get("content")):
                if b.get("type") == "tool_result":
                    out = b.get("content")
                    steps.append(base | {"step": "tool_result", "tool_use_id": b.get("tool_use_id"),
                                         "is_error": bool(b.get("is_error")), "chars": _size(out),
                                         "output": _preview(out, n)})
                elif b.get("type") == "text" and not e.get("isMeta"):
                    steps.append(base | {"step": "user_input", "chars": len(b.get("text", "")),
                                         "text": _preview(b.get("text"), n)})
        elif role == "assistant":
            usage = msg.get("usage") or {}
            for b in _blocks(msg.get("content")):
                t = b.get("type")
                if t == "thinking":
                    steps.append(base | {"step": "reasoning", "chars": len(b.get("thinking", "")),
                                         "text": _preview(b.get("thinking"), n)})
                elif t == "text":
                    steps.append(base | {"step": "assistant_text", "chars": len(b.get("text", "")),
                                         "text": _preview(b.get("text"), n)})
                elif t == "tool_use":
                    cat, mem = classify(b.get("name", ""), b.get("input"))
                    steps.append(base | {"step": "tool_call", "tool": b.get("name"), "tool_use_id": b.get("id"),
                                         "category": cat, "memory_command": mem, "input": _preview(b.get("input"), n)})
            if usage and steps:
                steps[-1]["usage"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens",
                                                                "cache_read_input_tokens",
                                                                "cache_creation_input_tokens") if k in usage}
                steps[-1]["model"] = msg.get("model")
    return steps, i if lines[start_line:] else start_line


def _flush_transcript(payload: dict, session: str, root, state: dict, why: str) -> None:
    tp = payload.get("transcript_path")
    if not tp:
        return
    steps, nxt = read_transcript(tp, state.get("transcript_line", 0), _max_chars(root))
    for s in steps:
        emit("trace", session, root, at=why, **s)
    state["transcript_line"] = nxt


# ---- the hook ----------------------------------------------------------------------------------------------------------
def hook(payload: dict) -> None:
    """Claude Code hook entry point for every event (`defrost hook monitor`). Never raises."""
    try:
        root = payload.get("cwd")
        if not enabled(root):
            return
        ev = payload.get("hook_event_name", "")
        session = payload.get("session_id") or "no-session"
        n = _max_chars(root)
        state = _load_state(session, root)
        if ev == "SessionStart":
            branch = subprocess.run(["git", "-C", str(_project_root(root)), "branch", "--show-current"],
                                    capture_output=True, text=True).stdout.strip()
            emit("session_start", session, root, source=payload.get("source"), model=payload.get("model"),
                 cwd=root, git_branch=branch or None, transcript=payload.get("transcript_path"))
        elif ev == "UserPromptSubmit":
            p = payload.get("prompt") or ""
            emit("prompt", session, root, chars=len(p), words=len(p.split()), text=_preview(p, n))
        elif ev == "PreToolUse":
            tool, ti = payload.get("tool_name", ""), payload.get("tool_input")
            cat, mem = classify(tool, ti)
            key = _tool_key(payload)
            state["starts"][key] = time.time()
            emit("tool_start", session, root, tool=tool, tool_use_id=key, category=cat, memory_command=mem,
                 input=_preview(ti, n))
        elif ev == "PostToolUse":
            tool, ti, resp = payload.get("tool_name", ""), payload.get("tool_input"), payload.get("tool_response")
            cat, mem = classify(tool, ti)
            key = _tool_key(payload)
            t0 = state["starts"].pop(key, None)
            emit("tool_end", session, root, tool=tool, tool_use_id=key, category=cat, memory_command=mem,
                 duration_ms=round((time.time() - t0) * 1000) if t0 else None, error=_is_error(resp),
                 output_chars=_size(resp), output=_preview(resp, n))
        elif ev in ("Stop", "SubagentStop", "PreCompact", "SessionEnd"):
            _flush_transcript(payload, session, root, state, ev)
            if ev == "SessionEnd":
                emit("session_end", session, root, reason=payload.get("reason"),
                     summary=summarize(load(_session_file(session, root))), open_tools=len(state["starts"]))
            else:
                emit({"Stop": "stop", "SubagentStop": "subagent_stop", "PreCompact": "pre_compact"}[ev], session, root,
                     trigger=payload.get("trigger"), open_tools=len(state["starts"]))
        elif ev == "Notification":
            emit("notification", session, root, message=_preview(payload.get("message"), n))
        else:
            emit("hook_event", session, root, name=ev)
        _save_state(session, state, root)
    except Exception as e:                                        # noqa: BLE001
        emit("monitor_error", payload.get("session_id"), payload.get("cwd"), error=f"{type(e).__name__}: {e}")


# ---- Claude Code wiring (project .claude/settings.json) --------------------------------------------------------------
def hook_command() -> str:
    """Shell-gated: python runs only when DEFROST_MONITOR is on (environment first, then the project's .env); always
    exits 0 with no output, so it can never block a tool or alter the conversation."""
    on = "1|true|on|yes|TRUE|ON|YES|True|On|Yes"
    off = "0|false|off|no|FALSE|OFF|NO|False|Off|No"
    env = '"${CLAUDE_PROJECT_DIR:-.}/.env"'
    pattern = r"""'^[[:space:]]*(export[[:space:]]+)?DEFROST_MONITOR[[:space:]]*=[[:space:]]*["'"'"']?(1|true|on|yes)'"""
    return (f"if command -v defrost >/dev/null 2>&1 && {{ case \"${{DEFROST_MONITOR:-}}\" in {on}) true;; {off}) false;; "
            f"*) grep -qsiE {pattern} {env};; esac; }}; then defrost hook monitor >/dev/null 2>&1 || true; fi  "
            f"# {HOOK_TAG}")


def _settings_file(root: Path) -> Path:
    return Path(root) / ".claude/settings.json"


def install_hook(root: Path) -> str:
    f = _settings_file(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(f.read_text()) if f.exists() and f.read_text().strip() else {}
    hooks = settings.setdefault("hooks", {})
    for ev, matcher in EVENTS.items():
        entries = [h for h in hooks.get(ev, []) if HOOK_TAG not in json.dumps(h)]
        entry = {"hooks": [{"type": "command", "command": hook_command(), "timeout": 10}]}
        if matcher:
            entry = {"matcher": matcher, **entry}
        hooks[ev] = entries + [entry]
    f.write_text(json.dumps(settings, indent=2) + "\n")
    return str(f)


def remove_hook(root: Path) -> None:
    f = _settings_file(root)
    if not f.exists():
        return
    settings = json.loads(f.read_text() or "{}")
    hooks = settings.get("hooks", {})
    for ev in list(hooks):
        kept = [h for h in hooks[ev] if HOOK_TAG not in json.dumps(h)]
        if kept:
            hooks[ev] = kept
        else:
            hooks.pop(ev)
    if "hooks" in settings and not hooks:
        settings.pop("hooks")
    f.write_text(json.dumps(settings, indent=2) + "\n")


def installed(root) -> bool:
    try:
        return HOOK_TAG in _settings_file(_project_root(root)).read_text()
    except OSError:
        return False


# ---- report ------------------------------------------------------------------------------------------------------------
def load(path: Path) -> list[dict]:
    out = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def summarize(events: list[dict]) -> dict:
    """Counts and reliability numbers for one session's events."""
    ends = [e for e in events if e["event"] == "tool_end"]
    starts = [e for e in events if e["event"] == "tool_start"]
    cats = Counter(e.get("category") for e in starts)
    mem_cmds = Counter(e.get("memory_command") for e in starts if e.get("memory_command"))
    dur = defaultdict(list)
    for e in ends:
        if e.get("duration_ms") is not None:
            dur[e.get("category")].append(e["duration_ms"])
    trace = [e for e in events if e["event"] == "trace"]
    usage = Counter()
    for e in trace:
        for k, v in (e.get("usage") or {}).items():
            usage[k] += v or 0
    injects = [e for e in events if e["event"] == "memory_inject"]
    t = [e["t"] for e in events if "t" in e]
    return {
        "events": len(events),
        "duration_s": round(max(t) - min(t), 1) if t else 0,
        "prompts": sum(e["event"] == "prompt" for e in events),
        "tool_calls": len(starts),
        "by_category": dict(cats),
        "grep_calls": cats.get("grep", 0),
        "memory_calls": cats.get("memory_search", 0) + cats.get("memory_update", 0),
        "memory_searches": cats.get("memory_search", 0),
        "memory_updates": cats.get("memory_update", 0),
        "memory_commands": dict(mem_cmds),
        "memory_injected": sum(e.get("decision") == "injected" for e in injects),
        "memory_inject_skipped": dict(Counter(e.get("reason") for e in injects if e.get("decision") != "injected")),
        "tool_errors": sum(bool(e.get("error")) for e in ends) + sum(
            e.get("step") == "tool_result" and bool(e.get("is_error")) for e in trace),
        "unfinished_tool_calls": max(0, len(starts) - len(ends)),
        "duration_ms_median": {c: sorted(v)[len(v) // 2] for c, v in dur.items()},
        "duration_ms_max": {c: max(v) for c, v in dur.items()},
        "reasoning_steps": sum(e.get("step") == "reasoning" for e in trace),
        "tokens": dict(usage),
        "compactions": sum(e["event"] == "pre_compact" for e in events),
        "subagents": sum(e["event"] == "subagent_stop" for e in events),
        "defrost_hook_errors": sum(e["event"] == "defrost_hook" and not e.get("ok", True) for e in events),
        "monitor_errors": sum(e["event"] == "monitor_error" for e in events),
    }


def sessions(root=None) -> list[Path]:
    d = log_dir(root)
    return sorted((p for p in d.glob("*.jsonl") if not p.name.startswith("_service-")),
                  key=lambda p: p.stat().st_mtime) if d.is_dir() else []


def sequence(events: list[dict]) -> list[str]:
    """The execution sequence, one line per step: user input -> memory injection -> reasoning -> tool call -> output.
    Live timings (tool_end) are attached to their tool result; the n-th memory injection follows the n-th input."""
    took = {e.get("tool_use_id"): e for e in events if e["event"] == "tool_end"}
    injects = [e for e in events if e["event"] == "memory_inject"]
    lines, n_input = [], 0
    for e in events:
        ev, s = e["event"], e.get("step")
        if ev == "trace":
            side = "  [subagent]" if e.get("sidechain") else ""
            if s == "user_input":
                lines.append(f"USER{side}      {e.get('text') or ''}"[:300])
                if not side and n_input < len(injects):
                    i = injects[n_input]
                    lines.append(f"  MEMORY-INJECT {i.get('decision')} {i.get('reason') or ''} best={i.get('best')} "
                                 f"{i.get('ms')}ms")
                    n_input += 1
            elif s == "reasoning":
                body = e.get("text") or "" if e.get("chars") else "(thinking not exposed in the transcript)"
                lines.append(f"  REASON{side}  ({e.get('chars')} chars) {body}"[:300])
            elif s == "assistant_text":
                lines.append(f"  SAY{side}     {e.get('text') or ''}"[:300])
            elif s == "tool_call":
                tag = e.get("memory_command") or e.get("category")
                lines.append(f"  TOOL{side}    {e.get('tool')} [{tag}] {e.get('input') or ''}"[:300])
            elif s == "tool_result":
                err = " ERROR" if e.get("is_error") else ""
                t = took.get(e.get("tool_use_id"), {}).get("duration_ms")
                ms = f", {t} ms" if t is not None else ""
                lines.append(f"  RESULT{side}{err} ({e.get('chars')} chars{ms}) {e.get('output') or ''}"[:300])
        elif ev in ("pre_compact", "subagent_stop", "session_end", "monitor_error"):
            lines.append(f"  -- {ev} {e.get('trigger') or e.get('reason') or e.get('error') or ''}")
    return [l.replace("\n", " ") for l in lines]


def report(root=None, last: int = 1, session: str | None = None, show_sequence: bool = False,
           as_json: bool = False) -> str:
    files = [_session_file(session, root)] if session else sessions(root)[-last:]
    out, data = [], []
    for f in files:
        ev = load(f)
        if not ev:
            continue
        s = summarize(ev)
        data.append({"session": f.stem, "file": str(f), "summary": s})
        if as_json:
            continue
        out.append(f"session {f.stem}  ({s['duration_s']} s, {s['prompts']} prompts, {s['tool_calls']} tool calls)")
        out.append(f"  grep calls: {s['grep_calls']}   memory calls: {s['memory_calls']} "
                   f"(search {s['memory_searches']}, update {s['memory_updates']})   "
                   f"memory injected into prompts: {s['memory_injected']}")
        if s["memory_commands"]:
            out.append("  memory commands: " + ", ".join(f"{k} x{v}" for k, v in s["memory_commands"].items()))
        if s["memory_inject_skipped"]:
            out.append("  injection skipped: " + ", ".join(f"{k} x{v}" for k, v in s["memory_inject_skipped"].items()))
        out.append("  tools by category: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_category"].items())))
        out.append(f"  errors: tool {s['tool_errors']}, unfinished {s['unfinished_tool_calls']}, defrost hooks "
                   f"{s['defrost_hook_errors']}, monitor {s['monitor_errors']}   compactions {s['compactions']}   "
                   f"subagents {s['subagents']}")
        if s["duration_ms_median"]:
            out.append("  median ms: " + ", ".join(f"{k} {v}" for k, v in s["duration_ms_median"].items()))
        if s["tokens"]:
            out.append("  tokens: " + ", ".join(f"{k} {v}" for k, v in s["tokens"].items()))
        if show_sequence:
            out.append("  sequence:")
            out.extend("    " + l for l in sequence(ev))
        out.append(f"  log: {f}")
    if as_json:
        return json.dumps(data, indent=1)
    return "\n".join(out) or f"no monitored sessions in {log_dir(root)}"


def status(root=None) -> str:
    r = _project_root(root)
    src = "environment" if os.environ.get("DEFROST_MONITOR") else (".env" if dotenv(r).get("DEFROST_MONITOR")
                                                                    else "not set")
    return (f"monitor: {'ON' if enabled(r) else 'off'} (DEFROST_MONITOR from {src})\n"
            f"hooks in {r / '.claude/settings.json'}: {'installed' if installed(r) else 'not installed'} "
            f"(`defrost setup . --monitor`)\n"
            f"logs: {log_dir(r)} ({len(sessions(r))} sessions)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="defrost monitor", description="Claude Code monitoring logs for this project")
    sub = ap.add_subparsers(dest="action", required=True)
    rp = sub.add_parser("report", help="counts, reliability and the execution sequence of recent sessions")
    rp.add_argument("--last", type=int, default=1)
    rp.add_argument("--session")
    rp.add_argument("--sequence", action="store_true", help="print user input / reasoning / tool / output steps")
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--project", default=None, help="project folder (default: current)")
    st = sub.add_parser("status", help="switch, hooks and log folder")
    st.add_argument("--project", default=None)
    a = ap.parse_args(argv)
    if a.action == "status":
        print(status(a.project))
    else:
        print(report(a.project, a.last, a.session, a.sequence, a.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
