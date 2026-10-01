"""Unit tests that need no model weights: parsing, links, retrieval policy, metrics, and a text-only build shape."""
import numpy as np

from kev_memory.evaluation.metrics import first_hit, ndcg_at_10, overlaps, paired_bootstrap
from kev_memory.ingest.documents import split_sections
from kev_memory.ingest.links import link_section, symbol_index
from kev_memory.retrieval import policy
from kev_memory.retrieval.keyword import fts_query


def test_markdown_sections_keep_heading_path_and_lines():
    md = "# Guide\nintro\n\n## Install\npip install x\n\n```\n# not a heading\n```\n## Use\nrun it\n"
    secs = split_sections(md, ".md")
    assert [s[1] for s in secs] == [["Guide"], ["Guide", "Install"], ["Guide", "Use"]]
    assert secs[1][2:4] == (4, 9)                 # the fenced '# not a heading' stays inside Install


def test_rst_and_asciidoc_headings_keep_line_numbers():
    rst = "Title\n=====\nintro\n\nSub\n---\ntext\n"
    assert [s[1] for s in split_sections(rst, ".rst")] == [["Title"], ["Title", "Sub"]]
    assert split_sections(rst, ".rst")[1][2] == 5
    adoc = "== Top\na\n=== Child\nb\n"
    assert [s[1] for s in split_sections(adoc, ".adoc")] == [["Top"], ["Top", "Child"]]


def test_heading_only_sections_are_skipped():
    md = "# Guide\n\n## Setup\n\n### Install\nrun it\n"
    assert [(s[1], s[2]) for s in split_sections(md, ".md")] == [(["Guide", "Setup", "Install"], 5)]


def test_doc_to_code_link_is_exact_and_unique():
    nodes = [{"id": "c_charge", "label": "charge_invoice()", "source_file": "c/billing.py", "component": "c"},
             {"id": "c_file", "label": "billing.py", "source_file": "c/billing.py", "component": "c"}]
    idx = symbol_index(nodes)
    links, unresolved, n, resolved = link_section("Use `charge_invoice` and `missing_fn()`.", idx,
                                                  {n["id"]: "c" for n in nodes}, "c")
    assert ("c_charge", "charge_invoice", "EXTRACTED", 1.0) in links
    assert unresolved == ["missing_fn()"]


def test_fast_policy_uses_hybrid_on_agreement_else_rerank():
    bm, de = ["a", "b", "c"], ["a", "c", "d"]
    assert not policy.needs_reranker("fast", bm, de)
    ranked, used = policy.rank("fast", bm, de)
    assert used == "hybrid" and ranked[0] == "a"
    bm2 = ["b", "a"]
    assert policy.needs_reranker("fast", bm2, de)
    ranked, used = policy.rank("fast", bm2, de, {"b": 0.1, "a": 2.0, "c": 1.0, "d": -1})
    assert used == "rerank" and ranked[:3] == ["a", "c", "b"]


def test_rrf_and_pool():
    assert policy.rrf([["x", "y"], ["y", "z"]])[0] == "y"
    assert policy.rerank_pool(list("abc"), list("cde")) == list("abcde")


def test_fts_query_splits_camel_case_and_drops_stop_words():
    assert fts_query("How does parseArgs work?") == '"parse" OR "args" OR "work"'


def test_metrics():
    assert overlaps("p", 10, 20, [{"path": "p", "lines": [15, 30]}])
    assert not overlaps("p", 10, 12, [{"path": "p", "lines": [15, 30]}])
    assert first_hit(["x", "g"], {"g"}) == 1 and ndcg_at_10(0) == 1.0 and ndcg_at_10(None) == 0.0
    d, lo, hi, p = paired_bootstrap(np.zeros(50), np.full(50, 0.1))
    assert abs(d - 0.1) < 1e-9 and lo > 0 and p == 0.0


def test_auto_k_sends_one_section_when_confident_else_more():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert policy.auto_k(ranked, {"a": 0.8, "b": 0.3, "c": 0.3, "d": 0.2}) == 1
    assert policy.auto_k(ranked, {"a": 0.50, "b": 0.49, "c": 0.48, "d": 0.2}) == 3
    assert policy.auto_k(ranked, {s: 0.4 for s in ranked + list("ghijk")}) == 5
    assert policy.auto_k(ranked, {}) == 5


def test_group_hits_counts_groups_found_in_top_k():
    from kev_memory.evaluation.metrics import group_hits
    ranked = ["x", "a1", "y", "z", "q", "w", "b2"]
    assert group_hits(ranked, [{"a1", "a2"}, {"b1", "b2"}], 5) == 1
    assert group_hits(ranked, [{"a1", "a2"}, {"b1", "b2"}], 10) == 2


def test_rerank_cache_scores_only_new_pairs_and_keeps_order():
    from collections import OrderedDict
    import numpy as np
    import pytest
    pytest.importorskip("torch")                           # CI runs without torch: skipped there
    from kev_memory.models.reranker import KevReranker
    rr = KevReranker.__new__(KevReranker)                 # no weights: the model call is replaced below
    rr.cache_size, rr._cache, calls = 3, OrderedDict(), []
    rr._score = lambda q, texts, batch_size=None: (calls.append(list(texts)), np.array([len(t) for t in texts], float))[1]
    assert list(rr.score("q", ["aa", "b"])) == [2, 1]
    assert list(rr.score("q", ["b", "ccc", "aa"])) == [1, 3, 2] and calls == [["aa", "b"], ["ccc"]]
    rr.score("q2", ["dddd"])                               # 4 entries > cache_size 3: the oldest is evicted
    assert len(rr._cache) == 3

def test_config_files_link_and_secrets_are_skipped():
    from kev_memory.config import SECRET_LIKE
    from kev_memory.ingest.links import mentions
    found = mentions("CI runs `.github/workflows/deploy.yml`; cron is in deploy/crontab and deploy/crm.Dockerfile.")[0]
    assert {".github/workflows/deploy.yml", "deploy/crontab", "deploy/crm.Dockerfile"} <= found
    assert all(SECRET_LIKE.search(p) for p in [".kamal/secrets", "deploy/.env", "certs/api.key", "pnpm-lock.yaml"])
    assert not any(SECRET_LIKE.search(p) for p in ["config/deploy.yml", "deploy/.env.example", "deploy/backup.sh"])


def test_missing_names_flag_only_paths_and_calls():
    from kev_memory.memory import PATH_OR_CALL
    assert PATH_OR_CALL.fullmatch("deploy/old_backup.sh") and PATH_OR_CALL.fullmatch("charge_invoice()")
    assert not PATH_OR_CALL.fullmatch("APP_DOMAIN") and not PATH_OR_CALL.fullmatch("SETUP.md")


def _graph():
    """A small graph: core backend file with a top-level function and a class method, a frontend file, a test file."""
    nodes = [
        {"id": "f_jobs", "label": "jobqueue.py", "source_file": "c/backend/jobqueue.py", "component": "c"},
        {"id": "pending", "label": "pending()", "source_file": "c/backend/jobqueue.py", "component": "c"},
        {"id": "JobQueue", "label": "JobQueue", "source_file": "c/backend/jobqueue.py", "component": "c"},
        {"id": "claim", "label": ".claim()", "source_file": "c/backend/jobqueue.py", "component": "c"},
        {"id": "f_ui", "label": "form.tsx", "source_file": "c/frontend/app/form.tsx", "component": "c"},
        {"id": "worker", "label": "worker()", "source_file": "c/frontend/app/form.tsx", "component": "c"},
        {"id": "f_t", "label": "jobs.test.ts", "source_file": "c/frontend/test/jobs.test.ts", "component": "c"},
        {"id": "query", "label": "query()", "source_file": "c/frontend/test/jobs.test.ts", "component": "c"},
        {"id": "f_s", "label": "sessions.py", "source_file": "c/backend/sessions.py", "component": "c"},
        {"id": "Session", "label": "Session", "source_file": "c/backend/sessions.py", "component": "c"},
    ]
    edges = [{"source": "f_jobs", "target": "pending", "relation": "contains"},
             {"source": "f_jobs", "target": "JobQueue", "relation": "contains"},
             {"source": "JobQueue", "target": "claim", "relation": "method"},
             {"source": "f_ui", "target": "worker", "relation": "contains"},
             {"source": "f_t", "target": "query", "relation": "contains"},
             {"source": "f_s", "target": "Session", "relation": "contains"}]
    return nodes, edges


def _extracted(body, nodes, edges, core=None):
    from kev_memory.ingest.links import symbol_index
    idx = symbol_index(nodes, edges)
    links, *_ = link_section(body, idx, {n["id"]: "c" for n in nodes}, "c", core)
    return [(n, m) for n, m, conf, _ in links if conf == "EXTRACTED"]


def test_bare_words_and_test_targets_do_not_link():
    nodes, edges = _graph()
    core = ["c/backend/"]
    got = _extracted("Status `pending` means review; the `worker` role; `rawPost.query` column.", nodes, edges, core)
    assert got == []                                   # status value, role, test helper: no link shown
    got = _extracted("In `jobqueue.py`, `pending` returns the backlog.", nodes, edges, core)
    assert ("pending", "pending") in got               # same section names the core file: the link stands


def test_methods_index_as_class_dot_method_and_dotted_fallback_is_checked():
    nodes, edges = _graph()
    assert ("claim", "JobQueue.claim") in _extracted("Call `JobQueue.claim` first.", nodes, edges)
    assert _extracted("The `data.claim` field.", nodes, edges) == []          # 'data' is not JobQueue / jobqueue
    assert ("claim", "jobqueue.claim") in _extracted("`jobqueue.claim` locks a row.", nodes, edges)
    assert ("Session", "Session") in _extracted("A `Session` holds cookies.", nodes, edges)  # class names stay


def test_core_paths_detection_and_display_order():
    from kev_memory.ingest.links import core_paths
    nodes, edges = _graph()
    assert core_paths({"components": []}, nodes) == ["c/backend/"]
    assert core_paths({"components": [{"name": "c", "core": ["frontend/app"]}]}, nodes) == ["c/frontend/app/"]
    got = _extracted("See `backend/jobqueue.py` and `JobQueue`, then `form.tsx`.", nodes, edges, ["c/backend/"])
    assert [n for n, _ in got] == ["f_jobs", "JobQueue", "f_ui"]             # core first, path before name


def test_doc_trust_config_round_trip_and_rule_text(tmp_path):
    import json
    from kev_memory import trust
    from kev_memory.project_setup import install_memory_rule
    ws = tmp_path / "w.workspace.json"
    ws.write_text(json.dumps({"name": "w", "components": []}))
    assert trust.read(ws) == "low"                     # default: code is the truth
    trust.write(ws, "high")
    assert trust.read(ws) == "high" and trust.read(None) == "low"
    install_memory_rule(tmp_path, "w", "high")
    text = (tmp_path / "CLAUDE.md").read_text()
    assert "doc trust: high" in text and text.count("defrost-ai:memory:start") == 1
    install_memory_rule(tmp_path, "w", "low")
    text = (tmp_path / "CLAUDE.md").read_text()
    assert "doc trust: low" in text and "doc trust: high" not in text
    try:
        trust.write(ws, "medium")
        raise AssertionError("invalid level accepted")
    except ValueError:
        pass


def test_context_starts_with_the_doc_trust_line():
    from kev_memory.memory import Memory
    hit = {"rank": 1, "domain": "w", "path": "w/a.md", "lines": [1, 2], "heading": "a.md > A", "text": "text",
           "code": [], "doc_trust": "high"}
    ctx = Memory.context({"hits": [hit]})
    assert ctx.startswith("[w] doc trust: HIGH")


def test_conflict_decisions_are_recorded_and_shown_on_matching_hits(tmp_path, monkeypatch):
    from kev_memory import conflicts
    from kev_memory.memory import Memory
    monkeypatch.setenv("KEV_MEMORY_HOME", str(tmp_path))
    conflicts.record("crm", "docs/DEPLOY.md", "code", doc_says="CI runs kamal deploy", code_does="CI SSHes to the server",
                     code_ref=".github/workflows/deploy.yml:12", doc_lines=[10, 20], note="CI only triggers the server")
    assert conflicts.for_section(conflicts.load("crm"), "docs/DEPLOY.md", [15, 30])
    assert not conflicts.for_section(conflicts.load("crm"), "docs/DEPLOY.md", [21, 30])
    hit = {"rank": 1, "domain": "crm", "path": "docs/DEPLOY.md", "lines": [12, 18], "heading": "Deploy > CI",
           "text": "CI runs kamal deploy.", "code": []}
    ctx = Memory.context({"hits": [hit]})
    assert "resolved: code is right: update the doc (user," in ctx
    import pytest
    with pytest.raises(ValueError):
        conflicts.record("crm", "docs/DEPLOY.md", "agent-guess")
