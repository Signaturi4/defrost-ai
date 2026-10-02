"""The 1.2 surface: two search modes, a settings file, 9 public commands, one hook entry point, 4 MCP tools."""
import json

import pytest


def test_settings_file_precedence_and_validation(tmp_path, monkeypatch):
    from defrost_ai import settings
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path))
    monkeypatch.delenv("DEFROST_SEARCH_MODE", raising=False)
    assert settings.get("search.mode") == "accurate" and settings.source("search.mode") == "default"
    settings.set_("search.mode", "fast")
    assert settings.get("search.mode") == "fast" and settings.source("search.mode") == "config.toml"
    text = (tmp_path / "config.toml").read_text()
    assert 'mode = "fast"' in text and "# " in text                       # commented, self-explaining file
    monkeypatch.setenv("DEFROST_SEARCH_MODE", "accurate")                  # env wins for one run
    assert settings.get("search.mode") == "accurate"
    with pytest.raises(ValueError):
        settings.set_("search.mode", "turbo")
    settings.set_("search.mode", None)
    monkeypatch.delenv("DEFROST_SEARCH_MODE")
    assert settings.get("search.mode") == "accurate"


def test_help_lists_only_the_public_commands(capsys):
    from defrost_ai import cli
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for c in cli.PUBLIC:
        assert f"    {c} " in out
    for hidden in ("docs-hook", "docs-record", "compact-handoff", "benchmark", "context", "hook "):
        assert f"    {hidden}" not in out


def test_old_commands_still_parse_for_installed_hooks():
    from defrost_ai import cli
    import argparse
    for argv in (["refresh", "acme", "--if-changed"], ["docs-pending"], ["brief"], ["docs-hook", "stop"],
                 ["setup", ".", "--on-main-merge", "--doc-rules", "--build", "skip"]):
        try:
            cli.main(argv + ["--help"])
        except SystemExit as e:
            assert e.code == 0, argv


def test_hook_entry_point_never_fails(monkeypatch, capsys, tmp_path):
    from defrost_ai import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("not json"))
    assert cli.main(["hook", "stop"]) == 0                                 # broken input: message, exit 0
    assert "defrost hook stop" in capsys.readouterr().err


def test_profiles_are_ordered_supersets():
    from defrost_ai.cli import PROFILES
    keys = [{k for k, v in PROFILES[p].items() if v} for p in ("minimal", "standard", "full")]
    assert keys[0] < keys[1] < keys[2]
    assert not PROFILES["standard"]["docs_gate"]                           # no commit blocking by default


def test_docs_hooks_without_gate_and_with_one_entry_point(tmp_path):
    import subprocess
    from defrost_ai import docsync
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    docsync.install_hooks(tmp_path, gate=False)
    hooks = json.loads((tmp_path / ".claude/settings.json").read_text())["hooks"]
    assert set(hooks) == {"SessionStart"} and "hook pending" in json.dumps(hooks)
    docsync.install_hooks(tmp_path, gate=True)                              # re-run replaces, never duplicates
    hooks = json.loads((tmp_path / ".claude/settings.json").read_text())["hooks"]
    assert set(hooks) == {"SessionStart", "Stop", "PreToolUse"} and len(hooks["SessionStart"]) == 1
    assert "hook post-commit" in (tmp_path / ".git/hooks/post-commit").read_text()


def test_mcp_server_has_four_tools():
    pytest.importorskip("mcp")
    import asyncio
    from defrost_ai.service.mcp_server import build_server
    tools = asyncio.run(build_server().list_tools())
    assert sorted(t.name for t in tools) == ["docs_for", "refresh", "remember", "search"]


def test_mcp_without_the_package_explains_instead_of_crashing(monkeypatch, capsys):
    from defrost_ai.service import mcp_server
    monkeypatch.setattr(mcp_server, "mcp_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        mcp_server.main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "`mcp` package" in err and "install.sh" in err and "connection closed" in err


def test_status_and_claude_install_flag_a_missing_mcp_package(monkeypatch, capsys, tmp_path):
    from defrost_ai import cli, project_setup
    from defrost_ai.integrations import claude
    from defrost_ai.service import mcp_server
    monkeypatch.setattr(mcp_server, "mcp_available", lambda: False)
    monkeypatch.setattr(project_setup, "status", lambda *a, **k: [])
    cli.main(["status"])
    assert "MCP server: NOT available" in capsys.readouterr().out
    monkeypatch.setattr(claude.shutil, "which", lambda name: None)       # no claude CLI: nothing is registered
    done = claude.install(tmp_path, register_mcp=True)
    assert any(d.startswith("WARNING: the `mcp` package is missing") for d in done)


def test_doc_plan_skips_paths_the_index_excludes(tmp_path, monkeypatch):
    from defrost_ai import docsync
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path))
    (tmp_path / "proj.workspace.json").write_text(json.dumps(
        {"components": [{"exclude": ["docs/tools/", "defrost-memory"]}]}))
    assert docsync._excluded("docs/tools/extract_facts.py", "proj")
    assert docsync._excluded("defrost-memory/notes/a.md", "proj")
    assert not docsync._excluded("docs/toolsmith.py", "proj")
    assert not docsync._excluded("src/app.py", "proj")
    assert not docsync._excluded("docs/tools/x.py", None)


def test_status_nests_the_notes_domain(monkeypatch, capsys):
    from defrost_ai import cli, project_setup
    rows = [{"domain": "app", "counts": {"sections": 5, "doc_code_links": 2}, "built_at": "2026-10-02T02:51",
             "stale": False, "triggers": []},
            {"domain": "app-context", "counts": {"docs": 2}, "built_at": None, "stale": True}]
    monkeypatch.setattr(project_setup, "status", lambda *a, **k: rows)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "app-context" not in out and "notes and decisions: 2 files" in out


def test_status_survives_a_deleted_workspace(tmp_path, monkeypatch):
    from defrost_ai import project_setup
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path))
    (tmp_path / "domains.json").write_text(json.dumps(
        {"domains": {"gone": {"workspace": str(tmp_path / "gone.workspace.json")}}}))
    rows = project_setup.status()
    assert rows[0]["domain"] == "gone" and rows[0]["stale"] and "missing" in rows[0]["why"]


def test_mcp_is_a_core_dependency_and_the_old_extra_still_resolves():
    import tomllib
    from pathlib import Path
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    assert any(d.replace(" ", "").startswith("mcp>=") for d in project["dependencies"])
    assert project["optional-dependencies"]["mcp"] == []          # `defrost-ai[mcp]` keeps resolving


def test_claude_hooks_are_portable_for_teammates(tmp_path, monkeypatch):
    """The committed .claude/settings.json must not carry this machine's path, must be a no-op where defrost is not
    installed, and must pass the hook's exit code through where it is (exit 2 = the docs gate holds a commit)."""
    import os
    import subprocess
    from pathlib import Path
    from defrost_ai import compact_handoff, docsync, notes
    monkeypatch.setattr(docsync, "git", lambda root, *a: str(tmp_path / ".git" / "hooks"))
    notes.install_hook(tmp_path)
    compact_handoff.install_hook(tmp_path)
    docsync.install_hooks(tmp_path, gate=True)
    text = (tmp_path / ".claude/settings.json").read_text()
    assert str(Path.home()) not in text and "/.local/bin" not in text
    cmds = [h["command"] for es in json.loads(text)["hooks"].values() for e in es for h in e["hooks"]]
    assert len(cmds) == 5 and all(c.startswith("if command -v defrost") for c in cmds)

    stop = next(c for c in cmds if "hook stop" in c)
    no_defrost = {"PATH": "/usr/bin:/bin"}
    assert subprocess.run(["sh", "-c", stop], env=no_defrost).returncode == 0
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "defrost").write_text("#!/bin/sh\nexit 2\n")
    (fake / "defrost").chmod(0o755)
    assert subprocess.run(["sh", "-c", stop], env={"PATH": f"{fake}{os.pathsep}/usr/bin:/bin"}).returncode == 2
