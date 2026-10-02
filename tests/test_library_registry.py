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
