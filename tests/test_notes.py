"""Session handoff notes: no models, no service (CI-safe)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from defrost_ai import notes
from defrost_ai.ingest.documents import split_sections


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path / "km"))
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (tmp_path / "km").mkdir()
    (tmp_path / "km" / "demo.workspace.json").write_text(json.dumps(
        {"name": "demo", "out": str(tmp_path / "km" / "demo"), "components": [{"name": "demo", "path": str(repo)}]}))
    return tmp_path, repo


def test_domain_for_finds_the_repo_domain(home):
    tmp, repo = home
    assert notes.domain_for(repo / "src") == "demo"
    assert notes.domain_for(tmp) is None


def test_handoff_note_is_sectioned_and_self_contained(home):
    f = notes.write_handoff("demo", "Make search under 3 s", "phase 2 done", ["bf16 for reranker only"],
                            ["run health check"], ["defrost_ai/models/reranker.py"], when=1_800_000_000)
    from defrost_ai.context_repo import parse
    secs = split_sections(parse(f.read_text())[1], ".md")             # body after the name/description frontmatter
    names = [s[1][-1] for s in secs]
    assert names[:2] == ["Goal (Make search under 3 s)", "State of the work (Make search under 3 s)"]
    assert all("Make search under 3 s" in n for n in names)          # every section names the goal


def test_brief_uses_newest_note_and_is_capped(home):
    assert notes.brief("demo") == ""
    notes.write_handoff("demo", "old goal", when=1_700_000_000)
    notes.write_handoff("demo", "new goal " + "word " * 600, "state", next_steps=["step one"], when=1_800_000_000)
    b = notes.brief("demo", max_words=50)
    assert "new goal" in b and "old goal" not in b
    assert len(b.split()) < 50 + 40 + 80 + 5                         # note cap + header/footer + map cap
    assert 'domains=["demo-context"]' in b


def test_register_notes_creates_docs_only_domain(home):
    tmp, _ = home
    ws = json.loads(notes.register_notes("demo").read_text())
    assert ws["name"] == "demo-context" and ws["components"][0]["path"].endswith("demo.context")
    reg = json.loads((tmp / "km" / "domains.json").read_text())
    assert "demo-context" in reg["domains"]
    assert notes.domain_for(notes.notes_dir("demo")) != "demo-context"


def test_hook_install_is_idempotent_and_removable(home):
    _, repo = home
    (repo / ".claude").mkdir()
    (repo / ".claude/settings.json").write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [
        {"type": "command", "command": "echo mine"}]}]}}))
    notes.install_hook(repo); notes.install_hook(repo)
    s = json.loads((repo / ".claude/settings.json").read_text())
    tagged = [h for h in s["hooks"]["SessionStart"] if notes.HOOK_TAG in json.dumps(h)]
    assert len(tagged) == 1 and tagged[0]["matcher"] == "clear|compact"
    notes.remove_hook(repo)
    s = json.loads((repo / ".claude/settings.json").read_text())
    assert s["hooks"]["SessionStart"] == [{"hooks": [{"type": "command", "command": "echo mine"}]}]


def test_brief_cli_is_fast_and_model_free(home):
    tmp, repo = home
    notes.write_handoff("demo", "goal for cli", next_steps=["next"])
    code = ("import sys; from defrost_ai.cli import main; main(['brief']); "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True,
                       env={**__import__("os").environ, "DEFROST_HOME": str(tmp / "km"),
                            "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert "goal for cli" in r.stdout
    assert r.stdout.strip().endswith("False False")
