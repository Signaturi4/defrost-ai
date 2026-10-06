"""`defrost forget NAME`: deletes one memory and nothing else."""
import json
import subprocess

import pytest


def _memory(home, name, repo, notes=None):
    out = home / name
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}")
    (home / f"{name}.previous").mkdir()
    ws = home / f"{name}.workspace.json"
    ws.write_text(json.dumps({"name": name, "out": str(out), "components": [{"name": name, "path": str(repo)}]}))
    (home / f"{name}.refresh.log").write_text("log")
    reg_file = home / "domains.json"
    reg = json.loads(reg_file.read_text()) if reg_file.exists() else {"domains": {}}
    reg["domains"][name] = {"workspace": str(ws), "description": ""}
    if notes:                                                  # working memory: its own repo, outside the home
        (home / f"{name}.context.path").write_text(str(notes))
        cws = home / f"{name}-context.workspace.json"
        cws.write_text(json.dumps({"name": f"{name}-context", "out": str(home / f"{name}-context"),
                                   "components": [{"name": f"{name}-context", "path": str(notes)}]}))
        (home / f"{name}-context").mkdir()
        reg["domains"][f"{name}-context"] = {"workspace": str(cws), "description": ""}
    reg_file.write_text(json.dumps(reg))


def _git(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    hook = path / ".git/hooks/post-commit"
    hook.write_text("#!/bin/sh\n# >>> defrost-ai >>>\necho refresh\n# <<< defrost-ai <<<\n")
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("DEFROST_HOME", str(h))
    return h


def test_forget_deletes_only_that_memory_and_keeps_the_notes(home, tmp_path):
    from defrost_ai.project_setup import forget
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "note.md").write_text("mine")
    _memory(home, "stray", tmp_path / "gone", notes)           # its repo was deleted (a /tmp worktree)
    _memory(home, "keep", _git(tmp_path / "keep"))
    plan = forget("stray")
    assert set(plan["unregister"]) == {"stray", "stray-context"}
    reg = json.loads((home / "domains.json").read_text())["domains"]
    assert list(reg) == ["keep"]
    assert not (home / "stray").exists() and not (home / "stray.previous").exists()
    assert not (home / "stray.workspace.json").exists() and not (home / "stray-context").exists()
    assert (home / "keep").exists() and (notes / "note.md").read_text() == "mine"


def test_hooks_shared_with_another_memory_are_kept(home, tmp_path):
    from defrost_ai.project_setup import forget
    repo = _git(tmp_path / "repo")
    _memory(home, "a", repo)
    _memory(home, "b", repo)                                   # two memories of one repository
    plan = forget("a")
    assert plan["remove_hooks_in"] == [] and plan["hooks_kept_shared_with_other_memories"] == [str(repo)]
    assert "defrost-ai" in (repo / ".git/hooks/post-commit").read_text()


def test_nothing_outside_the_home_is_deleted(home, tmp_path):
    from defrost_ai.project_setup import forget
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _memory(home, "odd", tmp_path / "gone")
    ws = home / "odd.workspace.json"
    ws.write_text(json.dumps({"name": "odd", "out": str(elsewhere), "components": []}))
    forget("odd")
    assert elsewhere.exists()


def test_unknown_name_and_dry_run(home, tmp_path):
    from defrost_ai.project_setup import forget
    with pytest.raises(KeyError):
        forget("nope")
    _memory(home, "x", tmp_path / "gone")
    plan = forget("x", dry_run=True)
    assert str(home / "x") in plan["delete"] and (home / "x").exists()


def test_cli_deletes_without_asking_and_dry_run_keeps(home, tmp_path):
    from defrost_ai import cli
    _memory(home, "x", tmp_path / "gone")
    assert cli.main(["forget", "x", "--dry-run"]) == 0 and (home / "x").exists()
    assert cli.main(["forget", "x"]) == 0 and not (home / "x").exists()
    assert cli.main(["forget", "x"]) == 1                         # already gone: says so, exit 1
