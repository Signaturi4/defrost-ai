"""Doc -> code link audit, offline (no models, no GPU): recompute every section's links from a built memory's
sections + code graph with the linker in this checkout, sample them by risk class for hand labelling, and score
labels before / after a linker change.

    python scripts/link_audit.py sample --memory ~/.defrost-ai/gcrm-grounding --n 50 --seed 1 >> sample.jsonl
    python scripts/link_audit.py show sample.jsonl            # doc sentence + target code, for labelling
    python scripts/link_audit.py score labels.jsonl           # precision per class; re-links with this checkout

Label rule ("correct"): the doc mention refers to that exact code entity (same function / class / file), so a reader
following the link lands on what the sentence talks about. "wrong": the mention means something else (a status value,
role, column, package, concept) or another entity with the same name."""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defrost_ai.ingest import links as L  # noqa: E402

TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec|e2e)(/|$)|[._-](test|spec)\.[a-z]+$|(^|/)test_[^/]+$")


def mention_class(m: str, how: str) -> str:
    if how == "dotted":
        return "dotted_fallback"
    if "/" in m or L.PATHLIKE.fullmatch(m):
        return "path"
    if re.fullmatch(r"[a-z]+", m):
        return "bare_lowercase"
    if re.fullmatch(r"[A-Z][a-z]+", m):
        return "capitalised"
    return "code_shaped"


def how_resolved(m: str, idx) -> str:   # classification of the BEFORE linker (sampling only)
    m = m.strip().strip("`'\"").rstrip(".,:;")
    m = m[:-2] if m.endswith("()") else m
    if idx.get(m):
        return "exact"
    if idx.get(m.split("(")[0]):
        return "paren"
    return "dotted"


def load_memory(mem: Path):
    db = sqlite3.connect(mem / "knowledge.sqlite")
    code = json.loads((mem / "code_graph.json").read_text())
    wsf = Path.home() / ".defrost-ai" / f"{mem.name}.workspace.json"
    if wsf.exists():
        ws = json.loads(wsf.read_text())
    else:                                   # test copies built from a temporary workspace: rebuild it from the manifest
        sources = list(json.loads((mem / "manifest.json").read_text()).get("sources", {}))
        names = sorted({c for (c,) in db.execute("SELECT DISTINCT component FROM sections")})
        ws = {"name": mem.name, "components": [{"name": n, "path": next((r for r in sources if Path(r).name == n),
                                                                          sources[0])} for n in names]}
    roots = {c["name"]: Path(c.get("path") or c["paths"][0]).expanduser() for c in ws["components"]}
    secs = db.execute("SELECT id, component, path, text FROM sections").fetchall()
    return db, code, roots, secs, ws


def relink(mem: Path):
    """-> {(sid, node_id): mention} of EXTRACTED links, computed by this checkout's linker."""
    db, code, roots, secs, ws = load_memory(mem)
    core = L.core_paths(ws, code["nodes"]) if hasattr(L, "core_paths") else None
    idx = L.symbol_index(code["nodes"], code.get("edges")) if hasattr(L, "SymbolIndex") else L.symbol_index(code["nodes"])
    comp = {n["id"]: n["component"] for n in code["nodes"]}
    out = {}
    for sid, component, path, text in secs:
        args = (core,) if hasattr(L, "core_paths") else ()
        links, *_ = L.link_section(text, idx, comp, component, *args)
        for nid, m, conf, _ in links:
            if conf == "EXTRACTED":
                out[(sid, nid)] = m
    return out, idx, code, roots, secs


def sample(a):
    mem = Path(a.memory).expanduser()
    links, idx, code, roots, secs = relink(mem)
    nodes = {n["id"]: n for n in code["nodes"]}
    by = defaultdict(list)
    for (sid, nid), m in links.items():
        cls = mention_class(m, how_resolved(m, idx))
        tgt_test = bool(TEST_PATH.search(nodes[nid].get("source_file") or ""))
        by["test_target" if tgt_test else cls].append((sid, nid, m, cls, tgt_test))
    quota = {"bare_lowercase": 12, "code_shaped": 12, "path": 8, "dotted_fallback": 8, "capitalised": 4, "test_target": 6}
    rnd = random.Random(a.seed)
    scale = a.n / sum(quota.values())
    for k, q in quota.items():
        pool = by.get(k, [])
        for sid, nid, m, cls, t in rnd.sample(pool, min(len(pool), round(q * scale))):
            print(json.dumps({"memory": mem.name, "section_id": sid, "node_id": nid, "mention": m, "class": cls,
                              "test_target": t, "stratum": k}))
    print(json.dumps({"_population": {k: len(v) for k, v in by.items()}, "memory": mem.name}), file=sys.stderr)


def show(a):
    cache = {}
    for i, line in enumerate(open(a.file)):
        r = json.loads(line)
        mem = Path("~/.defrost-ai").expanduser() / r["memory"]
        if mem not in cache:
            db, code, roots, secs, ws = load_memory(mem)
            cache[mem] = (db, {n["id"]: n for n in code["nodes"]}, roots)
        db, nodes, roots = cache[mem]
        path, text = db.execute("SELECT path, text FROM sections WHERE id=?", (r["section_id"],)).fetchone()
        m = r["mention"]
        j = text.find(m.split(".")[-1] if r["class"] == "dotted_fallback" else m)
        ctx = " ".join(text[max(0, j - 160): j + 160].split()) if j >= 0 else text[:300]
        n = nodes[r["node_id"]]
        comp, _, rel = (n.get("source_file") or "").partition("/")
        f = roots.get(comp, Path("/")) / rel
        loc = int((n.get("source_location") or "L1")[1:] or 1)
        try:
            lines = f.read_text(errors="replace").splitlines()
            snippet = " | ".join(x.strip() for x in lines[max(0, loc - 2): loc + 2])[:260]
        except OSError:
            snippet = "(file not readable)"
        print(f"#{i} [{r['stratum']}] mention `{m}`  doc {path}\n   doc: …{ctx}…\n   -> {n.get('label')} "
              f"{n.get('source_file')}:{n.get('source_location')}\n   code: {snippet}\n")


def score(a):
    rows = [json.loads(l) for l in open(a.file)]
    now = {}
    for mem in {r["memory"] for r in rows}:
        now[mem], *_ = relink(Path("~/.defrost-ai").expanduser() / mem)
    shown_before, shown_after = Counter(), Counter()
    tab = defaultdict(lambda: Counter())
    for r in rows:
        kept = (r["section_id"], r["node_id"]) in now[r["memory"]]
        k = r["stratum"]
        tab[k]["n"] += 1
        tab[k]["correct"] += r["label"] == "correct"
        tab[k]["kept"] += kept
        tab[k]["kept_correct"] += kept and r["label"] == "correct"
        tab[k]["removed_correct"] += (not kept) and r["label"] == "correct"
        tab[k]["removed_wrong"] += (not kept) and r["label"] == "wrong"
    tot = Counter()
    print(f"{'stratum':16s} {'n':>3s} {'prec before':>11s} {'kept':>5s} {'prec after':>10s} {'removed ok/wrong':>17s}")
    for k, c in sorted(tab.items()) + [("ALL", None)]:
        if c is None:
            c = tot
        else:
            tot.update(c)
        after = f"{c['kept_correct'] / c['kept']:.2f}" if c["kept"] else "-"
        print(f"{k:16s} {c['n']:3d} {c['correct'] / c['n']:11.2f} {c['kept']:5d} {after:>10s} "
              f"{c['removed_correct']:8d}/{c['removed_wrong']:<8d}")
    for mem, links in now.items():
        per = Counter(sid for sid, _ in links)
        print(f"{mem}: {len(links)} EXTRACTED links now; sections with links {len(per)}; "
              f"mean per linked section {sum(per.values()) / max(1, len(per)):.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--memory", required=True); s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=1)
    sh = sub.add_parser("show"); sh.add_argument("file")
    sc = sub.add_parser("score"); sc.add_argument("file")
    a = ap.parse_args()
    {"sample": sample, "show": show, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
