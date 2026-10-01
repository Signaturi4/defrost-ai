"""Doc sync: plan from a git diff + stored links, hooks, pending tasks. No models, no service (CI-safe)."""
import json
import subprocess

import pytest

from kev_memory import docsync, store


def sh(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    home = tmp_path / "km"
    monkeypatch.setenv("KEV_MEMORY_HOME", str(home))
    r = tmp_path / "proj"
    (r / "src").mkdir(parents=True); (r / "docs").mkdir(); (r / "deploy").mkdir()
    (r / "src/billing.py").write_text("def charge_invoice(x):\n    return x\n\n\ndef refund(x):\n    return -x\n")
    (r / "deploy/start.sh").write_text("#!/bin/sh\nexec app --port 8080\n")
    (r / "docs/BILLING.md").write_text("# Billing\n\n## Charging\nCall `charge_invoice()` to bill.\n\n"
                                       "## Refunds\nUse `refund()`.\n\n## Proxy\nThe `proxy` setting.\n")
    (r / "docs/DEPLOY.md").write_text("# Deploy\n\n## Start\nRun `deploy/start.sh`; it listens on `APP_PORT`.\n")
    sh(r, "git", "init", "-q"); sh(r, "git", "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    sh(r, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    out = home / "proj"; out.mkdir(parents=True)
    (home / "proj.workspace.json").write_text(json.dumps({"name": "proj", "out": str(out),
                                                          "components": [{"name": "proj", "path": str(r)}]}))
    (home / "domains.json").write_text(json.dumps({"domains": {"proj": {"workspace": str(home / "proj.workspace.json")}}}))
    nodes = [{"id": "f", "label": "billing.py", "source_file": "proj/src/billing.py", "source_location": "L1"},
             {"id": "c", "label": "charge_invoice()", "source_file": "proj/src/billing.py", "source_location": "L1"},
             {"id": "r", "label": "refund()", "source_file": "proj/src/billing.py", "source_location": "L5"},
             {"id": "p", "label": "proxy()", "source_file": "proj/src/billing.py", "source_location": "L5"}]
    (out / store.CODE_GRAPH).write_text(json.dumps({"nodes": nodes, "edges": []}))
    db = store.create(out / store.KNOWLEDGE_DB)
    secs = [("s1", "proj/docs/BILLING.md", "BILLING.md > Billing > Charging", 3, 4, "Call `charge_invoice()` to bill."),
            ("s2", "proj/docs/BILLING.md", "BILLING.md > Billing > Refunds", 6, 7, "Use `refund()`."),
            ("s3", "proj/docs/BILLING.md", "BILLING.md > Billing > Proxy", 9, 10, "The `proxy` setting."),
            ("s4", "proj/docs/DEPLOY.md", "DEPLOY.md > Deploy > Start", 3, 4,
             "Run `deploy/start.sh`; it listens on `APP_PORT`.")]
    for i, (sid, path, head, a, b, text) in enumerate(secs):
        db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, "d", "proj", path, head, 2, i, a, b, text, len(text.split()), 0, 0, 0))
    db.executemany("INSERT INTO links VALUES (?,?,?,?,?)", [("s1", "c", "charge_invoice()", "EXTRACTED", 1.0),
                                                           ("s2", "r", "refund()", "EXTRACTED", 1.0),
                                                           ("s3", "p", "proxy", "EXTRACTED", 1.0)])
    db.commit()
    return r


def test_plan_targets_changed_symbol_not_whole_file(repo):
    (repo / "src/billing.py").write_text("def charge_invoice(x):\n    return x\n\n\ndef refund(x):\n    return 0\n")
    p = docsync.plan(repo)
    first = [u["heading"] for u in p["update"] if u["priority"] == 0]
    assert first == ["BILLING.md > Billing > Refunds"]                  # refund() changed; charge_invoice() did not
    assert "BILLING.md > Billing > Proxy" not in [u["heading"] for u in p["update"]]   # bare word `proxy` ignored


def test_config_change_ranks_sections_naming_changed_text(repo):
    (repo / "deploy/start.sh").write_text("#!/bin/sh\nexec app --port \"$APP_PORT\"\n")
    p = docsync.plan(repo)
    u = [x for x in p["update"] if x["path"] == "docs/DEPLOY.md"][0]
    assert u["priority"] == 0 and any("APP_PORT" in r for r in u["reasons"])


def test_new_file_without_docs_is_undocumented_and_edited_docs_are_covered(repo):
    (repo / "src/access_code.py").write_text("def redeem(code):\n    return code\n")
    (repo / "src/billing.py").write_text("def charge_invoice(x):\n    return 2 * x\n\n\ndef refund(x):\n    return -x\n")
    (repo / "docs/BILLING.md").write_text((repo / "docs/BILLING.md").read_text() + "\nmore\n")
    p = docsync.plan(repo)
    assert [u["file"] for u in p["undocumented"]] == ["src/access_code.py"]
    assert p["covered_docs"] == ["docs/BILLING.md"] and not p["update"]       # its sections are not asked again


def test_stop_hook_blocks_once_per_change(repo):
    (repo / "src/billing.py").write_text("def charge_invoice(x):\n    return 3 * x\n\n\ndef refund(x):\n    return -x\n")
    first = docsync.hook("stop", {"cwd": str(repo)})
    assert first["decision"] == "block" and "Charging" in first["reason"]
    assert docsync.hook("stop", {"cwd": str(repo)}) is None                  # same change: not asked twice
    assert docsync.hook("stop", {"cwd": str(repo), "stop_hook_active": True}) is None


def test_commit_gate_only_for_git_commit_and_only_once(repo):
    (repo / "src/billing.py").write_text("def charge_invoice(x):\n    return 4 * x\n\n\ndef refund(x):\n    return -x\n")
    sh(repo, "git", "add", "src/billing.py")
    assert docsync.hook("commit", {"cwd": str(repo), "tool_input": {"command": "git status"}}) is None
    out = docsync.hook("commit", {"cwd": str(repo), "tool_input": {"command": "git add -A && git commit -m x"}})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert docsync.hook("commit", {"cwd": str(repo), "tool_input": {"command": "git commit -m x"}}) is None


def test_post_commit_records_pending_task(repo):
    (repo / "src/billing.py").write_text("def charge_invoice(x):\n    return 5 * x\n\n\ndef refund(x):\n    return -x\n")
    sh(repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "double charge")
    f = docsync.record_commit(repo)
    assert f and f.exists()
    assert "double charge" in docsync.pending_brief("proj")
    assert docsync.resolve("proj") == 1 and docsync.pending_brief("proj") == ""


def test_install_and_remove_hooks_keep_user_hooks(repo):
    (repo / ".claude").mkdir()
    (repo / ".claude/settings.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": "echo mine"}]}]}}))
    docsync.install_hooks(repo); docsync.install_hooks(repo, auto=True)
    s = json.loads((repo / ".claude/settings.json").read_text())
    assert len(s["hooks"]["Stop"]) == 2 and len(s["hooks"]["PreToolUse"]) == 1
    post = (repo / ".git/hooks/post-commit").read_text()
    assert post.count(docsync.MARK_START) == 1 and "CLAUDECODE" in post
    docsync.remove_hooks(repo)
    s = json.loads((repo / ".claude/settings.json").read_text())
    assert s == {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo mine"}]}]}}
    assert docsync.MARK_START not in (repo / ".git/hooks/post-commit").read_text()


def test_new_env_var_with_no_doc_is_reported(repo):
    (repo / "deploy/start.sh").write_text("#!/bin/sh\nexport MIGRATE_ON_START=1\nexec app --port 8080 --dry-run\n")
    p = docsync.plan(repo)
    assert p["new_names"] == [{"file": "deploy/start.sh", "names": ["--dry-run", "MIGRATE_ON_START"]}]
    assert docsync.hook("stop", {"cwd": str(repo)})["decision"] == "block"
