"""Grounding flags on a fake git repo (no model weights): stale needs changed lines that name the section's
identifiers, `verify in:` is capped at 3 with core/config files first, bare `Name.ext` found in code is no conflict."""
import os
import sqlite3
import subprocess
from collections import defaultdict

import pytest

from defrost_ai.memory import Memory

DOC_T = 1_700_000_000                                   # the doc's last commit
LATER = DOC_T + 86_400


def _git(root, *args, when=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"@{when} +0000"
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)


def _commit(root, path, text, when):
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", path, when=when)


def _memory(tmp_path, section_text, mentions=(), files=("app/jobs.py",), core=("c/app/",), config=()):
    """A Memory over a repo `c` whose doc section `s1` (docs/guide.md, committed at DOC_T) links to `files`."""
    root = tmp_path / "c"
    root.mkdir()
    _git(root, "init", "-q")
    _commit(root, "docs/guide.md", section_text, DOC_T)
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE sections(id, path, text, ordinal); CREATE TABLE docs(path, abspath, time);"
                     "CREATE TABLE links(section_id, node_id, mention, confidence, score);"
                     "CREATE TABLE unresolved(section_id, mention);")
    db.execute("INSERT INTO sections VALUES('s1', 'c/docs/guide.md', ?, 0)", (section_text,))
    db.execute("INSERT INTO docs VALUES('c/docs/guide.md', ?, ?)", (str(root / "docs/guide.md"), DOC_T))
    db.executemany("INSERT INTO unresolved VALUES('s1', ?)", [(m,) for m in mentions])
    m = object.__new__(Memory)
    m.db, m.workspace_file = db, None
    m.manifest = {"sources": {str(root): {}}, "core": list(core)}
    m.code_nodes, m.code_links, m.file_times = {}, defaultdict(list), {}
    for f in files:
        nid = "n_" + f
        m.code_nodes[nid] = {"id": nid, "source_file": "c/" + f, "kind": "config_file" if f in config else "code"}
        m.code_links["s1"].append(nid)
    return m, root


@pytest.fixture(autouse=True)
def _need_git():
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")


def test_stale_kept_when_changed_lines_name_the_section(tmp_path):
    m, root = _memory(tmp_path, "Jobs are claimed by `claim_next()` in `app/jobs.py`.")
    _commit(root, "app/jobs.py", "def claim_next():\n    pass\n", LATER)
    m.file_times["c/app/jobs.py"] = LATER
    _, stale = m.grounding("s1", "c/docs/guide.md")
    assert [s["file"] for s in stale] == ["c/app/jobs.py"] and "claim_next" in stale[0]["names"]


def test_stale_dropped_when_commit_touches_other_code(tmp_path):
    m, root = _memory(tmp_path, "Jobs are claimed by `claim_next()` in order.")
    _commit(root, "app/jobs.py", "def unrelated_helper():\n    return 1\n", LATER)
    m.file_times["c/app/jobs.py"] = LATER
    verify, stale = m.grounding("s1", "c/docs/guide.md")
    assert verify == ["c/app/jobs.py"] and stale == []  # newer file, but no name of the section in the diff


def test_plain_words_are_not_evidence(tmp_path):
    m, root = _memory(tmp_path, "The `deploy` step sets the `status` to done.")
    _commit(root, "app/jobs.py", "# deploy status\nstatus = 'deploy'\n", LATER)
    m.file_times["c/app/jobs.py"] = LATER
    assert m.section_names("s1") == set()
    assert m.grounding("s1", "c/docs/guide.md")[1] == []


def test_verify_capped_at_three_core_and_config_first(tmp_path):
    files = ("web/a.ts", "web/b.ts", "app/jobs.py", "web/c.ts", "deploy.yml")
    m, _ = _memory(tmp_path, "x", files=files, core=("c/app/",), config=("deploy.yml",))
    verify, _ = m.grounding("s1", "c/docs/guide.md")
    assert verify == ["c/app/jobs.py", "c/deploy.yml", "c/web/a.ts"]


def test_bare_filename_found_in_code_is_no_conflict(tmp_path):
    m, root = _memory(tmp_path, "Use `Prisma.sql` and see `rpa.js`.", mentions=("Prisma.sql", "rpa.js"))
    _commit(root, "app/q.ts", "const r = Prisma.sql`select 1`;\n", LATER)
    assert m.missing_names("s1", "c/docs/guide.md") == ["rpa.js"]
    _commit(root, "ext/rpa.js", "export {}\n", LATER)
    m.__dict__.pop("_tracked", None)
    assert m.missing_names("s1", "c/docs/guide.md") == []   # a tracked file with that name exists
