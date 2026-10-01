"""Training-data provenance guard (training/source/defrost_graph/bilm/data_provenance.py): allowlist, private
markers, 13-gram gate, stage manifests. Stdlib only: runs without torch and without any training data."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "data_provenance", ROOT / "training/source/defrost_graph/bilm/data_provenance.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)

REV = "a" * 40
MARKERS = {"path_patterns": ["acme-client"], "content_terms": ["acmecorp"], "private_corpora": []}


def spec(**over):
    s = {"schema": "defrost.sources/v1",
         "deny_path_patterns": ["(^|/)(Desktop|Documents|Downloads)(/|$)"],
         "eval_suites": [],
         "sources": [
             {"id": "oss:lib", "kind": "oss", "role": "train", "url": "https://github.com/o/lib", "revision": REV,
              "license": "MIT", "local_dir": "{oss}/lib", "aliases": ["oss/lib"]},
             {"id": "oss:held", "kind": "oss", "role": "heldout", "url": "https://github.com/o/held", "revision": REV,
              "license": "MIT", "local_dir": "{oss}/held"},
             {"id": "hf:set", "kind": "hf", "role": "train", "hf_id": "o/set", "revision": REV, "license": "MIT",
              "aliases": ["set"]}]}
    s.update(over)
    return s


@pytest.fixture
def clones(tmp_path):
    for repo in ("lib", "held"):
        (tmp_path / "oss" / repo / "docs").mkdir(parents=True)
        (tmp_path / "oss" / repo / "docs/index.md").write_text("# docs\n")
    return tmp_path


def allow(clones, markers=MARKERS, **over):
    return dp.Allowlist(spec(**over), {"oss": clones / "oss", "repo": clones}, markers)


# ── allowlist ────────────────────────────────────────────────────────────────

def test_listed_file_resolves_to_its_source(clones):
    a = allow(clones)
    assert a.source_for_path(clones / "oss/lib/docs/index.md")["id"] == "oss:lib"
    assert a.source("set")["id"] == "hf:set"                         # alias -> id


def test_unlisted_path_is_rejected(clones, tmp_path):
    other = tmp_path / "elsewhere/notes.md"
    other.parent.mkdir()
    other.write_text("x")
    with pytest.raises(dp.SourceRejected):
        allow(clones).source_for_path(other)
    with pytest.raises(dp.SourceRejected):
        allow(clones).source("hf:not-listed")


def test_local_home_folders_are_rejected(clones, tmp_path):
    doc = tmp_path / "Downloads/project/README.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("x")
    with pytest.raises(dp.PrivateDataError):
        allow(clones).source_for_path(doc)
    bad = spec()
    bad["sources"][0]["local_dir"] = "~/Documents/lib"
    with pytest.raises(ValueError, match="home folder"):
        dp.Allowlist(bad, {"oss": clones}, MARKERS)


def test_heldout_source_cannot_feed_training(clones):
    with pytest.raises(dp.SourceRejected, match="heldout"):
        allow(clones).source_for_path(clones / "oss/held/docs/index.md")
    assert allow(clones).source_for_path(clones / "oss/held/docs/index.md", roles=("heldout",))["id"] == "oss:held"


@pytest.mark.parametrize("field,value", [("revision", "main"), ("url", "http://github.com/o/lib"), ("license", "")])
def test_unpinned_or_unlicensed_sources_fail_validation(field, value):
    s = spec()
    s["sources"][0][field] = value
    assert dp.validate_spec(s)


def test_committed_allowlist_is_valid():
    s = json.loads((ROOT / "training/configs/sources_allowlist.json").read_text())
    assert dp.validate_spec(s) == []
    assert all(x.get("local_dir", "{").startswith("{") for x in s["sources"])     # clones under base dirs only


# ── denylist (private markers) ───────────────────────────────────────────────

def test_private_path_marker_fails_loudly(clones):
    with pytest.raises(dp.PrivateDataError):
        allow(clones).check_path("raw/oss/acme-client-docs/README.md")


def test_private_content_marker_fails_without_printing_it(clones):
    with pytest.raises(dp.PrivateDataError) as e:
        allow(clones).check_text("Deployed for AcmeCorp in 2025.", "doc.md")
    assert "acmecorp" not in str(e.value).lower()
    allow(clones).check_text("acmecorporation is a different word")          # whole words only


def test_strict_build_needs_the_markers_file(clones, tmp_path):
    with pytest.raises(dp.PrivateDataError):
        dp.Allowlist(spec(), {"oss": clones}, None, strict=True)
    dp.Allowlist(spec(), {"oss": clones}, None, strict=False)
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg/sources_allowlist.json").write_text(json.dumps(spec()))
    with pytest.raises(dp.PrivateDataError):
        dp.Allowlist.load(tmp_path / "cfg", strict=True)
    (tmp_path / "cfg/private_markers.local.json").write_text(json.dumps(MARKERS))
    assert dp.Allowlist.load(tmp_path / "cfg", strict=True).has_markers


# ── gate ─────────────────────────────────────────────────────────────────────

PRIVATE = ("the billing service retries every failed webhook three times before it pages the on call engineer "
           "and writes the failure to the audit table")
EVAL = ("to register a custom filter you add a function to the environment filters mapping under the name used "
        "in templates and then call it")
PUBLIC = "a public paragraph about parsing command line options with a decorator and typed parameters " * 2


def test_gate_fails_on_private_overlap_and_drops_eval_overlap():
    units = [("oss:lib", PUBLIC), ("oss:lib", "intro " + PRIVATE), ("hf:set", "x " + EVAL)]
    res = dp.gate(units, dp.grams(EVAL), dp.grams(PRIVATE))
    assert not res["ok"] and [i for i, _ in res["private_hits"]] == [1]
    assert [i for i, _ in res["eval_hits"]] == [2] and 2 not in res["kept"]
    clean = dp.gate([units[0], units[2]], dp.grams(EVAL), dp.grams(PRIVATE))
    assert clean["ok"] and clean["kept"] == [0]
    assert not dp.gate([units[0], units[2]], dp.grams(EVAL), set(), drop_eval=False)["ok"]


def test_boilerplate_shared_by_two_sources_does_not_count():
    units = [("oss:lib", PRIVATE), ("hf:set", PRIVATE)]
    assert dp.gate(units, set(), dp.grams(PRIVATE))["ok"]


def test_run_gate_raises_and_writes_a_private_report(clones, tmp_path):
    corpus = tmp_path / "private.txt"
    corpus.write_text(PRIVATE)
    a = allow(clones, markers=dict(MARKERS, private_corpora=[str(corpus)]))
    with pytest.raises(dp.PrivateDataError):
        dp.run_gate("unit", [("oss:lib", PRIVATE)], a, report_dir=tmp_path / "rep")
    report = json.loads((tmp_path / "rep/gate_unit.json").read_text())
    assert report["ok"] is False and report["private_hits_by_source"] == {"oss:lib": 1}
    assert PRIVATE[:30] not in json.dumps(report)                               # counts, never text


# ── manifest ─────────────────────────────────────────────────────────────────

def _stage(clones, tmp_path):
    f = tmp_path / "corpus_mntp.txt"
    units = [("oss:lib", PUBLIC), ("set", "a dataset sentence long enough to count")]
    f.write_text("\n".join(t for _, t in units) + "\n")
    a = allow(clones)
    res = dp.gate(units, set(), set())
    counts = {a.resolve(k): v for k, v in dp.count_units(units).items()}
    return a, f, dp.write_manifest("mntp_corpus", f, counts, a, res, inputs=[f])


def test_manifest_schema_round_trip(clones, tmp_path):
    a, f, m = _stage(clones, tmp_path)
    assert m.name == "corpus_mntp.txt.provenance.json"
    data = json.loads(m.read_text())
    assert data["schema"] == "defrost.provenance/v1" and data["sha256"] == dp.sha256(f)
    assert {s["id"] for s in data["sources"]} == {"oss:lib", "hf:set"}
    for s in data["sources"]:
        assert {"id", "kind", "revision", "license", "n_docs", "n_chars"} <= set(s)
    assert data["gate"]["ok"] and data["gate"]["private_hits"] == 0
    assert dp.verify_manifest(m, a) == []


def test_manifest_detects_tampering_and_unlisted_sources(clones, tmp_path):
    a, f, m = _stage(clones, tmp_path)
    f.write_text(f.read_text() + "one more line\n")
    assert "sha256 mismatch" in dp.verify_manifest(m, a)
    a, f, m = _stage(clones, tmp_path)
    data = json.loads(m.read_text())
    data["sources"].append(dict(data["sources"][0], id="local:Downloads"))
    data["gate"]["ok"] = False
    del data["created"]
    m.write_text(json.dumps(data))
    problems = " | ".join(dp.verify_manifest(m, a))
    assert "local:Downloads" in problems and "gate" in problems and "created" in problems


def test_write_manifest_refuses_unlisted_or_heldout_sources(clones, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    for sid in ("local/Downloads", "oss:held"):
        with pytest.raises(dp.SourceRejected):
            dp.write_manifest("s", f, {sid: {"n_docs": 1, "n_chars": 1}}, allow(clones))
