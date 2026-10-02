"""A registered memory whose workspace file was deleted must not break listing or all-memory search."""
import json

from defrost_ai import library


def test_stale_registration_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path / "home"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"built_at": "2026-10-02T00:00", "counts": {"sections": 1}}))
    ws = tmp_path / "good.workspace.json"
    ws.write_text(json.dumps({"name": "good", "out": str(out), "components": []}))
    library.register("good", ws)
    library.register("gone", tmp_path / "gone.workspace.json")          # workspace file never existed
    bad = tmp_path / "bad.workspace.json"
    bad.write_text("{}")                                                # malformed: no name/components
    library.register("bad", bad)

    doms = library.Library(models=object()).domains()

    assert doms["good"]["built"] is True
    for name in ("gone", "bad"):
        assert doms[name]["built"] is False and doms[name]["error"]
    # all-memory search picks only built domains, so the stale ones are skipped
    assert [n for n, d in doms.items() if d["built"]] == ["good"]


def test_fast_merge_ranks_across_domains_by_relevance_not_registry_order():
    """Each domain's #1 used to tie on rank, so the first-registered domains won whatever their relevance."""
    class FakeMemory:
        def __init__(self, cosines):
            self.cosines = cosines

        def search(self, query, mode, k, qvec):
            return {"k": 3, "mode_used": "hybrid", "hits": [
                {"rank": i + 1, "cosine": c, "heading": "", "text": "", "id": f"{id(self)}-{i}"}
                for i, c in enumerate(self.cosines)]}

    mems = {"weak": FakeMemory([0.15, 0.14, 0.13]),
            "strong": FakeMemory([0.64, 0.50, 0.70])}        # 3rd hit has a higher cosine but ranks 3rd in its domain

    class Models:
        class retriever:
            @staticmethod
            def embed_query(q):
                return None

    lib = library.Library(models=Models())
    lib.memory = lambda n: mems[n]
    out = lib.search("q", domains=["weak", "strong"], mode="fast", k=3, merge="rrf")
    assert [h["cosine"] for h in out["hits"]] == [0.64, 0.50, 0.70]   # all of "strong", in its own order
