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
