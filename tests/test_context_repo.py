"""Context repository (git-backed working memory, ported from Letta Code): CI-safe, no weights, no network."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from defrost_ai import context_constraints as cc


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DEFROST_DOMAIN", raising=False)
    return tmp_path / "home"


def git(cwd, *args, check=True):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if check:
        assert r.returncode == 0, r.stderr
    return r


# ---- validation rules (memory-frontmatter.ts / memory-constraints.ts) --------------------------------------------
def test_frontmatter_rules():
    ok = "---\nname: A\ndescription: about a\n---\nbody"
    assert cc.validate_frontmatter("notes/a.md", ok, None) == []
    assert "missing frontmatter" in cc.validate_frontmatter("a.md", "body", None)[0]
    assert "must not have frontmatter" in cc.validate_frontmatter("notes/MEMORY.md", ok, None)[0]
    assert any("unknown frontmatter key 'type'" in e
               for e in cc.validate_frontmatter("a.md", "---\nname: A\ndescription: d\ntype: x\n---\n", None))
    assert any("'name' must not be empty" in e for e in cc.validate_frontmatter("a.md", "---\nname:\ndescription: d\n---\n", None))
    ro = "---\nname: A\ndescription: d\nread_only: true\n---\nold"
    assert "read_only" in cc.validate_frontmatter("a.md", ro.replace("old", "new"), ro)[0]
    assert any("cannot be set" in e for e in cc.validate_frontmatter("a.md", ro, None))


def test_tree_constraints_and_config():
    files = [("MEMORY.md", "100644"), ("core.md", "100644"), ("notes/a.md", "100644"), ("x/y/z/deep.md", "100644")]
    text = {"MEMORY.md": "map", "core.md": "c" * 50, "notes/a.md": "n" * 30, "x/y/z/deep.md": "d"}
    errors = cc.validate_tree(files, text.__getitem__, {"version": 1, "maxDepth": 2, "maxFileCharacters": 40,
                                                       "maxCoreMemoryCharacters": 45,
                                                       "fileCharacterLimits": [{"pattern": "notes/*.md",
                                                                                "maxCharacters": 20}]})
    joined = "\n".join(errors)
    assert "missing required index notes/MEMORY.md" in joined and "x/MEMORY.md" in joined
    assert "depth 3 exceeds maxDepth 2" in joined
    assert "core.md: 50 characters exceeds 40" in joined and "glob 'notes/*.md'" in joined
    assert "core memory: 53 characters exceeds 45" in joined
    with pytest.raises(ValueError):
        cc.parse_config('{"version": 2, "bogus": 1}')
    assert cc.glob_regex("notes/**/x*.md").match("notes/a/b/x1.md")


# ---- repository, hook, commits, indexes ------------------------------------------------------------------------
def test_handoff_commit_index_brief_and_facts(home):
    from defrost_ai import context_repo as cr, notes
    f = notes.write_handoff("crm", "Ship the CSV export", "API done", ["use streaming"], ["add UI button"],
                            ["backend/export.py"], when=1_700_000_000)
    root = cr.repo_dir("crm")
    assert f.parent == root / "notes" and f.exists()
    meta, body = cr.parse(f.read_text())
    assert meta["name"].startswith("Handoff") and meta["description"].startswith("Ship the CSV export")
    assert "Facts:\n- Ship the CSV export → next step → add UI button" in body
    log = cr.log("crm")
    assert log[0]["subject"] == "handoff: Ship the CSV export" and log[0]["author"] == "defrost-ai"
    assert log[-1]["subject"].startswith("init:")
    assert f.name in (root / "notes/MEMORY.md").read_text()
    assert "[notes/](notes/MEMORY.md)" in (root / "MEMORY.md").read_text()
    assert cr.check("crm") == []
    b = notes.brief("crm")
    assert "Ship the CSV export" in b and "add UI button" in b and "[context repository map]" in b


def test_pre_commit_hook_rejects_bad_files_and_protected_config(home):
    from defrost_ai import context_repo as cr
    root = cr.ensure("crm")
    (root / "notes/bad.md").write_text("no frontmatter here\n")
    git(root, "add", "-A")
    r = git(root, "commit", "-m", "bad", check=False)
    assert r.returncode != 0 and "missing frontmatter" in r.stderr
    git(root, "reset", "-q", "--hard")
    (root / "notes/bad.md").unlink(missing_ok=True)
    (root / cc.CONFIG_PATH).write_text(json.dumps({"version": 1, "maxFileCharacters": 10**9}))
    git(root, "add", "-A")
    r = git(root, "commit", "-m", "raise limits", check=False)
    assert r.returncode != 0 and "is protected" in r.stderr
    git(root, "reset", "-q", "--hard")
    with pytest.raises(cr.ContextError):                      # programmatic writes are validated and rolled back
        cr.commit("crm", {"notes/x.md": "no frontmatter"}, "bad write")
    assert not (root / "notes/x.md").exists() and git(root, "status", "--porcelain").stdout == ""


def test_conflict_decisions_live_in_the_repo_and_migrate_from_jsonl(home):
    from defrost_ai import conflicts, context_repo as cr
    home.mkdir(parents=True)
    old = {"at": "2026-09-01T10:00:00", "domain": "crm", "doc_path": "docs/A.md", "doc_lines": [1, 5], "doc_says": "x",
           "code_ref": "a.py:1", "code_does": "y", "decision": "doc", "meaning": conflicts.DECISIONS["doc"], "note": ""}
    (home / "crm.conflicts.jsonl").write_text(json.dumps(old) + "\n")
    (home / "crm-notes/notes").mkdir(parents=True)
    (home / "crm-notes/notes/20260901-100000-old.md").write_text("# Handoff old\n\n## Goal (old)\nold goal\n")
    assert [r["doc_path"] for r in conflicts.load("crm")] == ["docs/A.md"]          # readable before migration
    row = conflicts.record("crm", "docs/B.md", "code", "CI runs kamal", "CI ssh", ".github/workflows/d.yml:3", [10, 20])
    rows = conflicts.load("crm")
    assert [r["doc_path"] for r in rows] == ["docs/A.md", "docs/B.md"] and rows[1]["file"] == row["file"]
    assert (home / "crm.conflicts.jsonl.migrated").exists() and (home / "crm-notes/notes.migrated").exists()
    assert (cr.repo_dir("crm") / "notes/20260901-100000-old.md").exists()
    subjects = [r["subject"] for r in cr.log("crm")]
    assert any(s.startswith("migrate:") for s in subjects) and subjects[0].startswith("decision(code): docs/B.md")
    assert conflicts.for_section(rows, "docs/B.md", [15, 16])
    text = (cr.repo_dir("crm") / row["file"]).read_text()
    assert "→ user decision → code is right" in text and cr.check("crm") == []


# ---- worktree jobs (memory-worktree.ts) -------------------------------------------------------------------------
def _repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "a.txt").write_text("one\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "init")
    return r


def test_worktree_job_fast_forwards_and_never_touches_the_checkout(home, tmp_path):
    from defrost_ai import worktree as wt
    repo = _repo(tmp_path)
    (repo / "dirty.txt").write_text("user's uncommitted work\n")
    res = wt.run(repo, "docs", lambda d: (Path(d) / "doc.md").write_text("new doc\n"), "docs: add")
    assert res["status"] == "merged" and (repo / "doc.md").read_text() == "new doc\n"
    assert (repo / "dirty.txt").exists() and not wt.branches(repo)
    assert wt.run(repo, "noop", lambda d: None, "nothing")["status"] == "no_changes"
    kept = wt.run(repo, "docs", lambda d: (Path(d) / "b.md").write_text("b\n"), "docs: b", merge=False,
                  branch="defrost/docs/abc123")
    assert kept["status"] == "kept" and not (repo / "b.md").exists()
    assert [b["branch"] for b in wt.branches(repo)] == ["defrost/docs/abc123"]
    assert wt.merge_branch(repo, "defrost/docs/abc123")["status"] == "merged" and (repo / "b.md").exists()


def test_worktree_conflict_keeps_the_branch(home, tmp_path):
    from defrost_ai import worktree as wt
    repo = _repo(tmp_path)
    kept = wt.run(repo, "edit", lambda d: (Path(d) / "a.txt").write_text("worker\n"), "worker edit", merge=False)
    (repo / "a.txt").write_text("user\n")
    git(repo, "commit", "-qam", "user edit")                  # main moved on: not a fast-forward any more
    res = wt.merge_branch(repo, kept["branch"])
    assert res["status"] == "merge_conflict" and (repo / "a.txt").read_text() == "user\n"
    assert [b["branch"] for b in wt.branches(repo)] == [kept["branch"]]
    assert wt.run(repo, "boom", lambda d: 1 / 0, "never")["status"] == "failed"
    assert len(wt.branches(repo)) == 1                        # a failed job leaves no branch behind


def test_defrag_archives_old_notes_and_splits_oversize_files(home):
    from defrost_ai import context_repo as cr, notes
    for i in range(5):
        notes.write_handoff("crm", f"task {i}", when=1_700_000_000 + i * 60)
    big = "\n".join(f"## Part {i}\n" + "word " * 300 for i in range(20))  # ~30k chars: above the 20k limit
    root = cr.repo_dir("crm")
    (root / "notes/zz-big.md").write_text(cr.render("Big", "a long imported note", big))
    git(root, "add", "-A"); git(root, "commit", "-q", "--no-verify", "-m", "import an oversize note")
    assert any("exceeds 20000" in e for e in cr.check("crm"))
    res = cr.defrag("crm", keep_notes=2)
    assert res["status"] == "merged" and res["archived"] >= 3 and res["split"] == 1
    assert len([f for f in (root / "notes/archive").glob("*.md") if f.name != "MEMORY.md"]) == res["archived"]
    assert list((root / "notes").glob("zz-big-part*.md")) and cr.check("crm") == []
    assert cr.log("crm")[0]["subject"].startswith(f"defrag: archived {res['archived']} note(s), split 1 file(s)")


# ---- extractive handoff before compaction -----------------------------------------------------------------------
def test_compact_handoff_extracts_goal_todos_files_questions(home, tmp_path, monkeypatch):
    from defrost_ai import compact_handoff, context_repo as cr
    lines = [
        {"type": "user", "message": {"content": "Add rate limiting to the export API"}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "backend/api.py"}},
            {"type": "tool_use", "name": "TodoWrite", "input": {"todos": [
                {"content": "limiter middleware", "status": "completed"},
                {"content": "tests for 429", "status": "in_progress"},
                {"content": "document RATE_LIMIT env var", "status": "pending"}]}},
            {"type": "text", "text": "Done with the middleware.\nShould the limit be per org or per user?"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}},
        {"type": "user", "message": {"content": "<command-name>/compact</command-name>"}},
    ]
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join(json.dumps(x) for x in lines))
    x = compact_handoff.extract(t)
    assert x["goal"] == "Add rate limiting to the export API" and x["files"] == ["backend/api.py"]
    assert x["next_steps"] == ["document RATE_LIMIT env var"] and "1 done, 1 in progress, 1 pending" in x["state"]
    assert x["decisions"] == ["open question: Should the limit be per org or per user?"]
    monkeypatch.setenv("DEFROST_DOMAIN", "crm")
    f = compact_handoff.run({"transcript_path": str(t), "cwd": str(tmp_path), "trigger": "auto"})
    assert f and f.exists() and "before auto compaction" in cr.parse(f.read_text())[0]["description"]


def test_map_brief_respects_the_word_budget(home):
    from defrost_ai import context_repo as cr
    cr.ensure("crm")
    cr.commit("crm", {"conventions.md": cr.render("Conventions", "standing rules", "rule " * 2000)}, "core facts")
    b = cr.map_brief("crm", max_words=120)
    assert len(b.split(" ")) <= 121 and b.endswith("…") and "Context repository: crm" in b


def test_memory_files_follow_the_writing_rules_except_frontmatter_type(home):
    """docs/WRITING_FOR_EXTRACTION.md via doc_lint: no ERROR other than the doc-page frontmatter `type` (memory
    files use Letta's name/description frontmatter, enforced by the pre-commit hook instead)."""
    import importlib.util
    from defrost_ai import conflicts, notes
    spec = importlib.util.spec_from_file_location(
        "doc_lint", Path(__file__).resolve().parents[1] / "defrost_ai/assets/doc_rules/tools/doc_lint.py")
    lint = importlib.util.module_from_spec(spec); spec.loader.exec_module(lint)
    note = notes.write_handoff("crm", "Ship the CSV export", "done", [], ["add UI"], ["backend/export.py"])
    dec = conflicts.record("crm", "docs/B.md", "code", "CI runs kamal", "CI ssh", "ci.yml:3", [1, 2])
    from defrost_ai import context_repo as cr
    for f in (note, cr.repo_dir("crm") / dec["file"]):
        errors = [i for i in lint.lint(f) if i[0] == "ERROR" and "frontmatter type" not in i[2]]
        assert errors == [], (f, errors)


def test_place_moves_the_repo_into_the_project_and_hides_it_from_project_git(home, tmp_path):
    from defrost_ai import context_repo as cr, notes
    proj = tmp_path / "proj"
    proj.mkdir()
    git(proj, "init", "-q")
    notes.write_handoff("p", "goal before the move", "", [], [], [])          # created under the home
    assert cr.repo_dir("p") == home / "p.context"
    res = cr.place("p", proj)
    assert res["moved"] and cr.repo_dir("p") == proj / "defrost-memory" and not (home / "p.context").exists()
    assert "goal before the move" in notes.latest("p").read_text()
    assert "/defrost-memory/" in (proj / ".git/info/exclude").read_text()
    assert git(proj, "status", "--porcelain").stdout == ""                    # the project's git does not see it
    notes.write_handoff("p", "goal after the move", "", [], [], [])
    assert cr.check("p") == [] and len(cr.log("p")) >= 3
    cr.place("p", None)                                                       # back to the home
    assert cr.repo_dir("p") == home / "p.context" and (home / "p.context/.git").exists()
