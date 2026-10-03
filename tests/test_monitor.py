"""Monitoring mode: off by default, switched in .env, never blocks, safe to delete, logs what the report counts."""
import io
import json
import os
import subprocess
import sys

import pytest

from defrost_ai import monitor


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setenv("DEFROST_MONITOR_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("DEFROST_MONITOR", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    return root


def _on(root, value="on"):
    (root / ".env").write_text(f"# local\nOTHER=1\nDEFROST_MONITOR={value}\n")


# ---- switch -------------------------------------------------------------------------------------------------------
def test_off_by_default_on_from_dotenv_and_env_wins(project, monkeypatch):
    assert not monitor.enabled(project)
    _on(project, '"yes"  # quoted, commented')
    assert monitor.enabled(project) and monitor.enabled(project / "sub/dir")      # found from a subfolder
    monkeypatch.setenv("DEFROST_MONITOR", "off")
    assert not monitor.enabled(project)


def test_nothing_is_written_when_off(project):
    monitor.hook({"hook_event_name": "UserPromptSubmit", "session_id": "s1", "cwd": str(project), "prompt": "hi"})
    assert not monitor.log_dir(project).exists()


# ---- classification ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tool,inp,expected", [
    ("Grep", {"pattern": "x"}, ("grep", None)),
    ("Bash", {"command": "grep -rn artem ."}, ("grep", None)),
    ("Bash", {"command": "cd x && rg foo"}, ("grep", None)),
    ("Bash", {"command": "ls -la"}, ("shell", None)),
    ("Bash", {"command": "defrost search 'who is artem' --fast"}, ("memory_search", "cli:defrost search")),
    ("Bash", {"command": "~/.local/bin/defrost refresh"}, ("memory_update", "cli:defrost refresh")),
    ("mcp__defrost__search", {"question": "q"}, ("memory_search", "mcp:search")),
    ("mcp__defrost__remember", {"kind": "note"}, ("memory_update", "mcp:remember")),
    ("mcp__defrost__refresh", {}, ("memory_update", "mcp:refresh")),
    ("Read", {}, ("read", None)), ("Edit", {}, ("edit", None)), ("mcp__figma__x", {}, ("mcp", None)),
])
def test_classify(tool, inp, expected):
    assert monitor.classify(tool, inp) == expected


# ---- a whole session ----------------------------------------------------------------------------------------------
def _transcript(path):
    rows = [
        {"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "who is artem"}},
        {"type": "assistant", "timestamp": "t2", "message": {"model": "claude-sonnet-5", "content": [
            {"type": "thinking", "thinking": "search the memory first"},
            {"type": "tool_use", "id": "tu1", "name": "mcp__defrost__search", "input": {"question": "who is artem"}}],
            "usage": {"input_tokens": 100, "output_tokens": 20}}},
        {"type": "user", "timestamp": "t3", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "[1] agency:a.md:L1-6 Artem"}]}},
        {"type": "assistant", "timestamp": "t4", "message": {"content": [
            {"type": "tool_use", "id": "tu2", "name": "Bash", "input": {"command": "grep -rn artem ."}}],
            "usage": {"input_tokens": 50, "output_tokens": 5}}},
        {"type": "user", "timestamp": "t5", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu2", "content": "boom", "is_error": True}]}},
        {"type": "assistant", "timestamp": "t6", "message": {"content": [{"type": "text", "text": "Artem is ..."}]}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_session_is_logged_counted_and_sequenced(project, tmp_path):
    _on(project)
    tr = tmp_path / "t.jsonl"
    _transcript(tr)
    base = {"session_id": "s1", "cwd": str(project), "transcript_path": str(tr)}
    calls = [
        {"hook_event_name": "SessionStart", "source": "startup"},
        {"hook_event_name": "UserPromptSubmit", "prompt": "who is artem"},
        {"hook_event_name": "PreToolUse", "tool_name": "mcp__defrost__search", "tool_use_id": "tu1",
         "tool_input": {"question": "who is artem"}},
        {"hook_event_name": "PostToolUse", "tool_name": "mcp__defrost__search", "tool_use_id": "tu1",
         "tool_input": {"question": "who is artem"}, "tool_response": "[1] agency:a.md"},
        {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_use_id": "tu2",
         "tool_input": {"command": "grep -rn artem ."}},
        {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_use_id": "tu2",
         "tool_input": {"command": "grep -rn artem ."}, "tool_response": {"stdout": "", "exit_code": 1}},
        {"hook_event_name": "PreToolUse", "tool_name": "mcp__defrost__remember", "tool_use_id": "tu3",
         "tool_input": {"kind": "note"}},
        {"hook_event_name": "Stop"},
        {"hook_event_name": "Stop"},                                               # nothing new: no duplicate trace
        {"hook_event_name": "SessionEnd", "reason": "exit"},
    ]
    for c in calls:
        monitor.hook(base | c)
    events = monitor.load(monitor._session_file("s1", project))
    s = [e for e in events if e["event"] == "session_end"][0]["summary"]
    assert s["prompts"] == 1 and s["tool_calls"] == 3
    assert s["grep_calls"] == 1 and s["memory_searches"] == 1 and s["memory_updates"] == 1
    assert s["memory_commands"] == {"mcp:search": 1, "mcp:remember": 1}
    assert s["unfinished_tool_calls"] == 1                                        # remember never finished
    assert s["tool_errors"] == 2                                                  # grep exit 1 + transcript is_error
    assert s["tokens"] == {"input_tokens": 150, "output_tokens": 25} and s["reasoning_steps"] == 1
    steps = [e["step"] for e in events if e["event"] == "trace"]
    assert steps == ["user_input", "reasoning", "tool_call", "tool_result", "tool_call", "tool_result",
                     "assistant_text"]
    rep = monitor.report(project, show_sequence=True)
    assert "grep calls: 1" in rep and "memory calls: 2" in rep and "mcp:search x1" in rep
    seq = "\n".join(monitor.sequence(events))
    assert seq.index("USER") < seq.index("REASON") < seq.index("TOOL") < seq.index("RESULT") < seq.index("SAY")
    assert "RESULT ERROR" in seq


def test_previews_can_be_turned_off(project, monkeypatch):
    _on(project)
    monkeypatch.setenv("DEFROST_MONITOR_MAX_CHARS", "0")
    monitor.hook({"hook_event_name": "UserPromptSubmit", "session_id": "s2", "cwd": str(project),
                  "prompt": "secret client name"})
    rec = monitor.load(monitor._session_file("s2", project))[0]
    assert rec["chars"] == 18 and rec["text"] is None


def test_internal_events_go_to_the_session_or_the_service_log(project):
    _on(project)
    from defrost_ai import monitor_event
    monitor_event("memory_inject", "s3", str(project), decision="skipped", reason="below relevance threshold")
    os.chdir(project)
    monitor_event("mcp_tool", None, None, tool="search", ms=12, ok=True)
    files = sorted(p.name for p in monitor.log_dir(project).glob("*.jsonl"))
    assert files[0].startswith("_service-") and files[1] == "s3.jsonl"


# ---- the hook command: shell-gated, never blocks -----------------------------------------------------------------
@pytest.fixture
def fake_defrost(tmp_path):
    """A `defrost` on PATH that records it ran and exits 2 (a blocking exit code for PreToolUse)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    marker = tmp_path / "ran"
    (bin_ / "defrost").write_text(f"#!/bin/sh\ncat >/dev/null\necho ran >> {marker}\necho noise\nexit 2\n")
    (bin_ / "defrost").chmod(0o755)
    return bin_, marker


def _run_hook(project, bin_, env_value=None):
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "CLAUDE_PROJECT_DIR": str(project)}
    if env_value is not None:
        env["DEFROST_MONITOR"] = env_value
    return subprocess.run(["bash", "-c", monitor.hook_command()], input="{}", capture_output=True, text=True,
                          env=env, cwd=project)


def test_hook_command_is_gated_silent_and_never_blocks(project, fake_defrost):
    bin_, marker = fake_defrost
    r = _run_hook(project, bin_)
    assert r.returncode == 0 and not marker.exists()                              # off: python never starts
    _on(project)
    r = _run_hook(project, bin_)
    assert r.returncode == 0 and r.stdout == "" and marker.read_text().count("ran") == 1   # exit 2 swallowed
    r = _run_hook(project, bin_, env_value="off")
    assert r.returncode == 0 and marker.read_text().count("ran") == 1             # env off beats .env on
    (project / ".env").unlink()
    _run_hook(project, bin_, env_value="1")
    assert marker.read_text().count("ran") == 2


def test_hook_command_without_defrost_installed(project):
    _on(project)
    r = subprocess.run(["bash", "-c", monitor.hook_command()], input="{}", capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "CLAUDE_PROJECT_DIR": str(project)})
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""


def test_install_is_idempotent_keeps_other_hooks_and_removes_cleanly(project):
    f = project / ".claude/settings.json"
    f.parent.mkdir()
    f.write_text(json.dumps({"model": "sonnet", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x"}]}]}}))
    monitor.install_hook(project)
    monitor.install_hook(project)
    hooks = json.loads(f.read_text())["hooks"]
    assert set(hooks) == set(monitor.EVENTS)
    assert len(hooks["Stop"]) == 2 and hooks["Stop"][0]["hooks"][0]["command"] == "x"
    assert hooks["PreToolUse"][0]["matcher"] == "*" and "/Users/" not in json.dumps(hooks)
    monitor.remove_hook(project)
    s = json.loads(f.read_text())
    assert s == {"model": "sonnet", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x"}]}]}}


# ---- deleting the module leaves everything working -------------------------------------------------------------------
def test_everything_works_without_the_monitor_module(project, monkeypatch, capsys):
    import defrost_ai
    monkeypatch.setitem(sys.modules, "defrost_ai.monitor", None)                  # import now raises ImportError
    monkeypatch.delattr(defrost_ai, "monitor")                                   # as if the file were deleted
    _on(project)
    from defrost_ai import cli, monitor_event, project_setup
    monitor_event("x", "s", str(project))                                         # no-op
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name": "Stop"}'))
    assert cli.main(["hook", "monitor"]) == 0
    assert project_setup._monitor_hooks(project, True) is False
    assert cli.main(["monitor", "status"]) == 1 and "not installed" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert cli.main(["hook", "stop"]) == 0                                        # other hooks unaffected


def test_cli_status_and_report(project, capsys, monkeypatch):
    from defrost_ai import cli
    _on(project)
    monitor.hook({"hook_event_name": "UserPromptSubmit", "session_id": "s9", "cwd": str(project), "prompt": "who"})
    assert cli.main(["monitor", "status", "--project", str(project)]) == 0
    out = capsys.readouterr().out
    assert "monitor: ON (DEFROST_MONITOR from .env)" in out and "1 sessions" in out
    assert cli.main(["monitor", "report", "--project", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["summary"]["prompts"] == 1


def test_prompt_hook_and_mcp_tools_report_to_the_monitor(project, monkeypatch):
    pytest.importorskip("mcp")
    _on(project)
    seen = []
    import defrost_ai
    monkeypatch.setattr(defrost_ai, "monitor_event", lambda ev, s=None, r=None, **f: seen.append((ev, f)))
    from defrost_ai import prompt_context
    prompt_context.hook({"prompt": "ok", "session_id": "s", "cwd": str(project)})
    assert seen[-1][0] == "memory_inject" and seen[-1][1]["reason"] == "command or short prompt"
    from defrost_ai.service import mcp_server
    import asyncio
    monkeypatch.setattr(mcp_server, "_here", lambda: None)
    asyncio.run(mcp_server.build_server().call_tool("remember", {"kind": "note", "goal": "g"}))
    assert seen[-1][0] == "mcp_tool" and seen[-1][1]["tool"] == "remember" and "ms" in seen[-1][1]
