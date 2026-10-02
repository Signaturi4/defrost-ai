"""The UserPromptSubmit hook (answer lookups in one turn) and the mandatory doc-trust question in setup."""
import json
import subprocess

import pytest

from defrost_ai import prompt_context


# ---- which prompts are searched ----------------------------------------------------------------------------------
@pytest.mark.parametrize("prompt,expected", [
    ("who is artem", True), ("what is our hourly rate", True),
    ("ok", False), ("continue", False), ("yes do it", True), ("git push", False), ("", False),
    ("/clear", False), ("/defrost-setup now please", False),
])
def test_only_real_questions_are_searched(prompt, expected):
    assert prompt_context._worth_searching(prompt) is expected


def _fake_service(monkeypatch, cosine, *, alive=True, context="doc trust: HIGH\n\n[1] agency:a.md:L1-5  Artem"):
    from defrost_ai import notes
    from defrost_ai.service import client
    calls = []

    def call(method, path, body=None, timeout=600):
        calls.append((method, path, body, timeout))
        if path == "/domains":
            return {"agency": {"built": True}, "agency-context": {"built": False}}
        return {"hits": [{"cosine": cosine}], "context": context}
    monkeypatch.setattr(notes, "domain_for", lambda p=".": "agency")
    monkeypatch.setattr(client, "alive", lambda: alive)
    monkeypatch.setattr(client, "_call", call)
    monkeypatch.delenv("DEFROST_PROMPT_MIN_COSINE", raising=False)
    return calls


def test_relevant_question_gets_the_sections_in_one_fast_search(monkeypatch):
    calls = _fake_service(monkeypatch, 0.42)
    out = prompt_context.hook({"prompt": "who is artem", "cwd": "."})
    ctx = out["hookSpecificOutput"]
    assert ctx["hookEventName"] == "UserPromptSubmit"
    assert "agency:a.md:L1-5" in ctx["additionalContext"] and "without searching again" in ctx["additionalContext"]
    search = [c for c in calls if c[1] == "/search"][0]
    assert search[2]["mode"] == "fast" and search[2]["k"] == 3 and search[2]["domains"] == ["agency"]   # built only
    assert search[3] <= 5                                                    # bounded: never hangs the prompt


def test_unrelated_prompt_adds_nothing(monkeypatch):
    _fake_service(monkeypatch, 0.20)
    assert prompt_context.hook({"prompt": "refactor this function to be async"}) is None


def test_threshold_is_a_setting(monkeypatch):
    _fake_service(monkeypatch, 0.42)
    monkeypatch.setenv("DEFROST_PROMPT_MIN_COSINE", "0.5")
    assert prompt_context.hook({"prompt": "who is artem"}) is None


def test_cold_service_is_started_in_the_background_not_awaited(monkeypatch):
    calls = _fake_service(monkeypatch, 0.9, alive=False)
    started = []
    monkeypatch.setattr(prompt_context, "_start_service_in_background", lambda: started.append(1))
    assert prompt_context.hook({"prompt": "who is artem"}) is None
    assert started == [1] and calls == []


def test_prompt_hook_entry_point_never_fails(monkeypatch, capsys):
    from defrost_ai import cli
    monkeypatch.setattr(prompt_context, "context_for", lambda p: 1 / 0)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO('{"prompt": "who is artem"}'))
    assert cli.main(["hook", "prompt"]) == 0
    out = capsys.readouterr()
    assert out.out == "" and "defrost hook prompt" in out.err                # the prompt goes through unchanged


def test_install_is_portable_idempotent_and_removable(tmp_path):
    f = tmp_path / ".claude/settings.json"
    f.parent.mkdir()
    f.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "mine"}]}]}}))
    prompt_context.install_hook(tmp_path)
    prompt_context.install_hook(tmp_path)
    entries = json.loads(f.read_text())["hooks"]["UserPromptSubmit"]
    assert len(entries) == 2 and entries[0]["hooks"][0]["command"] == "mine"            # the user's hook is kept
    cmd = entries[1]["hooks"][0]
    assert cmd["command"].startswith("if command -v defrost") and "hook prompt" in cmd["command"]
    assert "/Users/" not in cmd["command"] and cmd["timeout"] <= 10
    prompt_context.remove_hook(tmp_path)
    assert json.loads(f.read_text())["hooks"]["UserPromptSubmit"] == entries[:1]


def test_portable_command_is_a_noop_without_defrost(tmp_path):
    prompt_context.install_hook(tmp_path)
    cmd = json.loads((tmp_path / ".claude/settings.json").read_text())["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    r = subprocess.run(["bash", "-c", cmd["command"]], input='{"prompt":"who is artem"}', capture_output=True,
                       text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and r.stdout == ""


# ---- doc trust: suggested from the repository, required once, kept on re-runs -------------------------------------
def _repo(tmp_path, docs, code):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for i in range(docs):
        (tmp_path / f"d{i}.md").write_text("# doc\n")
    for i in range(code):
        (tmp_path / f"c{i}.py").write_text("x = 1\n")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/tool.py").write_text("")                           # tooling, not the project's code
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    return tmp_path


def test_suggest_trust_high_for_a_docs_repository_low_for_code(tmp_path):
    from defrost_ai import trust
    assert trust.suggest(_repo(tmp_path / "docs", 30, 2))[0] == "high"
    assert trust.suggest(_repo(tmp_path / "nocode", 3, 0))[0] == "high"
    assert trust.suggest(_repo(tmp_path / "code", 10, 5))[0] == "low"


def _setup_cli(monkeypatch, tmp_path):
    from defrost_ai import cli, project_setup
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DEFROST_DOC_TRUST", raising=False)
    monkeypatch.setattr(project_setup, "HOME", tmp_path / "home")
    seen = []
    monkeypatch.setattr(project_setup, "setup", lambda *a, **k: seen.append(k) or {"domain": "r", "triggers": {}})
    monkeypatch.setattr(cli, "_print_setup", lambda *a: None)
    return cli, seen


def test_yes_without_doc_trust_stops_and_suggests_for_a_new_project(monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path / "r", 20, 0)
    cli, seen = _setup_cli(monkeypatch, tmp_path)
    assert cli.main(["setup", str(repo), "--yes", "--build", "skip"]) == 2
    err = capsys.readouterr().err
    assert "--doc-trust high" in err and "Suggested here: --doc-trust high" in err and seen == []


def test_doc_trust_given_or_stored_is_used_and_rerun_keeps_it(monkeypatch, tmp_path):
    repo = _repo(tmp_path / "r", 20, 0)
    cli, seen = _setup_cli(monkeypatch, tmp_path)
    assert cli.main(["setup", str(repo), "--yes", "--doc-trust", "high", "--build", "skip"]) == 0
    assert seen[-1]["doc_trust"] == "high" and seen[-1]["prompt_context"] is True       # standard profile
    (tmp_path / "home").mkdir()
    (tmp_path / "home/r.workspace.json").write_text(json.dumps({"name": "r", "doc_trust": "high"}))
    assert cli.main(["setup", str(repo), "--yes", "--build", "skip", "--no-prompt-context"]) == 0
    assert seen[-1]["doc_trust"] is None and seen[-1]["prompt_context"] is False         # None = keep "high"


def test_suggest_trust_flag_changes_nothing(monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path / "r", 20, 0)
    cli, seen = _setup_cli(monkeypatch, tmp_path)
    assert cli.main(["setup", str(repo), "--suggest-trust"]) == 0
    assert json.loads(capsys.readouterr().out) == {"suggested": "high", "stored": None,
                                                   "why": "20 doc files and no code: the docs are the source of truth"}
    assert seen == []


def test_setup_installs_and_a_rerun_without_it_removes_the_prompt_hook(tmp_path, monkeypatch):
    from defrost_ai import project_setup
    monkeypatch.setattr(project_setup, "HOME", tmp_path / "home")
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "r", 3, 0)
    monkeypatch.setattr("defrost_ai.library.register", lambda *a, **k: None)
    project_setup.setup(str(repo), "r", build="skip", doc_trust="high", prompt_context=True, memory_dir=None)
    hooks = json.loads((repo / ".claude/settings.json").read_text())["hooks"]
    assert "hook prompt" in json.dumps(hooks["UserPromptSubmit"])
    project_setup.setup(str(repo), "r", build="skip", prompt_context=False, memory_dir=None)
    hooks = json.loads((repo / ".claude/settings.json").read_text()).get("hooks", {})
    assert "UserPromptSubmit" not in hooks
    from defrost_ai import trust
    assert trust.read(tmp_path / "home/r.workspace.json") == "high"              # re-run without doc_trust keeps it
