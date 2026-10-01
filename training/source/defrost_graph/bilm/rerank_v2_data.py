"""Defrost-Rerank v2 training data: the v1 groups with harder negatives, plus long-form prose.

The v1 reranker errs in two known ways (docs/RESULTS.md, diagnostics): it promotes CHANGELOG / "upgrading" sections,
and it promotes sibling sections of the right file with generic headings. v1 never saw either as a negative:
changelog files were skipped when sections were sampled, and BM25 negatives came from the whole project. It also
trails bge-reranker-v2-m3 on books (0.862 vs 0.949): long narrative prose is outside its training mix.

  prose      open-licensed books -> sections -> one reader question per section (Claude Haiku)
               train: The Rust Programming Language, Rust by Example, the Rustonomicon (MIT/Apache-2.0),
                      The Hitchhiker's Guide to Python (CC BY-NC-SA 3.0), System Design Primer (CC BY 4.0)
               dev:   Mostly Adequate Guide to FP (CC BY-SA 4.0), never trained on -> defrost_graph/memory/evals/prose_dev_v1.jsonl
  changelog  changelog / release-note sections as positives (questions by Claude Haiku)
  build      groups {source, query, positive, negatives[7]}:
               techdoc  up to 2 same-file siblings + up to 2 changelog/release-note sections (same project first) +
                        v1 BM25 negatives
               prose    up to 2 same-chapter siblings + BM25 over the same book
               public   v1 groups unchanged (replay against forgetting)
             every new negative passes the same 13-gram leakage gate as v1 (all eval suites + their KBs).

    python -m defrost_graph.bilm.rerank_v2_data prose                 # prose sections + questions (resumable)
    python -m defrost_graph.bilm.rerank_v2_data changelog_questions   # changelog sections as positives (resumable)
    python -m defrost_graph.bilm.rerank_v2_data candidates            # candidate groups, leakage-gated
    ../defrost/.venv/bin/python -m defrost_graph.bilm.rerank_v2_filter   # Defrost-Ret-B false-negative filter
    python -m defrost_graph.bilm.rerank_v2_data build                 # -> defrost_graph/data/rerank_v2/{rerank_train,rerank_val}.jsonl"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from defrost_graph.bilm import data_provenance as dp
from defrost_graph.bilm.sup_data import (DOCS, EXCLUDE, OUT as SUP, SCHEMA, TECHDOC_INSTRUCTION, allowed_train_doc, allowlist,
                                     eval_grams, fts, gate_and_write, grams13, shingles)

OUT = Path("defrost_graph/data/rerank_v2")
PROSE = Path("defrost_graph/data/prose")
REPOS = PROSE / "repos"
BOOKS = {   # name: (repo dir, glob, license, role)
    "rust-book": ("book", "src/**/*.md", "MIT OR Apache-2.0", "train"),
    "rust-by-example": ("rust-by-example", "src/**/*.md", "MIT OR Apache-2.0", "train"),
    "nomicon": ("nomicon", "src/**/*.md", "MIT OR Apache-2.0", "train"),
    "python-guide": ("python-guide", "docs/**/*.rst", "CC BY-NC-SA 3.0", "train"),
    "system-design-primer": ("system-design-primer", "README.md", "CC BY 4.0", "train"),
    "mostly-adequate": ("mostly-adequate-guide", "*.md", "CC BY-SA 4.0", "dev"),
}
SKIP_FILES = re.compile(r"(SUMMARY|README|CONTRIBUTING|LICENSE|CODE_OF_CONDUCT|TRANSLAT|FAQ)\.", re.I)
CHANGELOG = re.compile(r"(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS|WHATSNEW|RELEASE[-_ ]?NOTES|UPGRAD\w*)[^/]*$|/releases?/", re.I)
PROSE_PROMPT = """You write training questions for a search engine over technical books.
For EACH numbered book section below, write ONE question a reader of the book would realistically ask whose answer is
in that section: a how/why/what-happens question in plain words, the way someone who half-remembers the idea would ask.
Paraphrase: do not copy the heading or distinctive multi-word phrases. The question must be specific to that section.
Set usable=false for sections with no real content (navigation, link lists, exercises without explanation).

{sections}"""
CHANGELOG_PROMPT = """You write training questions for a documentation search engine used by developers.
Each numbered section below comes from a project's changelog or release notes. For EACH section, write ONE question a
developer would realistically ask whose answer is in that section: when something was added, removed, deprecated or
changed, what changed about a feature, or why an upgrade broke something. Paraphrase: do not copy distinctive
multi-word phrases (identifiers are fine). The question must be specific to that section. Set usable=false for
sections with no real content (version headers only, link lists, contributor lists).

{sections}"""
N_NEG = 7
MAX_CL_POS = 1000                       # changelog sections used as positives
CL_POS_PER_PROJECT = 300
FILTER_MARGIN = 0.95                    # drop a new negative whose Defrost-Ret-B cosine >= 0.95 x the positive's
MAX_SIB = 2
MAX_CHANGELOG = 2


def claude(prompt):
    cmd = ["claude", "-p", prompt, "--model", "haiku", "--output-format", "json", "--tools", "", "--json-schema",
           json.dumps(SCHEMA), "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    for _ in range(4):
        try:
            out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                            stdin=subprocess.DEVNULL).stdout)
            if isinstance(out, list):
                out = next(e for e in reversed(out) if e.get("type") == "result")
            return (out.get("structured_output") or json.loads(out["result"]))["items"]
        except Exception:
            continue
    return None


def split_doc(text, path):
    from defrost_graph.memory.text_kb import sections_of
    return [(" > ".join(hp), a, b, t) for _, hp, a, b, t in sections_of(text, rst=path.endswith(".rst"))]


# ---- prose -------------------------------------------------------------------------------------------------------
def prose_sections():
    from defrost_graph.memory.text_bench import code_share
    rows = []
    for book, (repo, pat, lic, role) in BOOKS.items():
        root = REPOS / repo
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        src = allowlist().source(book, roles=("train", "dev"))
        if commit != src["revision"] or src["role"] != role or src["license"] != lic:
            raise dp.SourceRejected(f"{book}: clone / role / license differ from the allowlist entry {src['id']}")
        for f in sorted(root.glob(pat)):
            rel = str(f.relative_to(root))
            if SKIP_FILES.search(f.name) and not (book == "system-design-primer" and f.name == "README.md"):
                continue
            secs = split_doc(f.read_text(errors="ignore"), rel)
            for j, (head, a, b, t) in enumerate(secs):
                nw = len(t.split())
                if 60 <= nw <= 400 and code_share(t) < 0.5:
                    rows.append({"book": book, "source_id": src["id"], "role": role, "license": lic, "commit": commit,
                                 "path": rel,
                                 "chapter": f"{book}/{rel}", "k": j, "heading": head, "lines": [a, b], "text": t})
    print(Counter((r["book"], r["role"]) for r in rows))
    return rows


def prose(per_call=8, workers=12):
    PROSE.mkdir(parents=True, exist_ok=True)
    f_secs, f_q = PROSE / "sections.jsonl", PROSE / "questions.jsonl"
    if not f_secs.exists():
        f_secs.write_text("".join(json.dumps(r) + "\n" for r in prose_sections()))
    secs = [json.loads(l) for l in open(f_secs)]
    done = {json.loads(l)["i"] for l in open(f_q)} if f_q.exists() else set()
    batches = [list(range(b, min(b + per_call, len(secs)))) for b in range(0, len(secs), per_call)]
    batches = [b for b in batches if not all(i in done for i in b)]
    print(f"{len(secs)} sections, {len(batches)} calls to go", flush=True)

    def run(b):
        body = "\n\n".join(f"[{k}] ({secs[i]['book']}) {secs[i]['heading']}\n{secs[i]['text'][:2500]}"
                           for k, i in enumerate(b))
        items = claude(PROSE_PROMPT.format(sections=body)) or []
        return [{"i": b[it["n"]], "question": it["question"].strip()} for it in items
                if 0 <= it.get("n", -1) < len(b) and it.get("usable") and it.get("question", "").strip()]

    with ThreadPoolExecutor(workers) as ex, open(f_q, "a") as f:
        for k, rows in enumerate(ex.map(run, batches)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(batches)} calls", flush=True)
    dev_suite()


def dev_suite():
    """The held-out book -> a frozen dev suite in the text_bench format (gold = the source section's line span)."""
    secs = [json.loads(l) for l in open(PROSE / "sections.jsonl")]
    qs = [json.loads(l) for l in open(PROSE / "questions.jsonl")]
    rows = []
    for q in sorted(qs, key=lambda q: q["i"]):
        s = secs[q["i"]]
        if s["role"] != "dev":
            continue
        rows.append({"id": f"prose-{len(rows):03d}", "split": "dev", "component": s["book"], "multi": False,
                     "question": q["question"], "answer": "", "gold": [{"path": f"{s['book']}/{s['path']}", "lines": s["lines"]}]})
    f = Path("defrost_graph/memory/evals/prose_dev_v1.jsonl")
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"dev suite {f}: {len(rows)} questions, sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:12]}")


# ---- groups --------------------------------------------------------------------------------------------------------
def near_copy(a, b):
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / max(1, len(sa | sb)) > 0.5


def bm25_order(db, query, n=30):
    from defrost_graph.memory.retrieve import fts_query
    fq = fts_query(query)
    return [rid for (rid,) in db.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?", (fq, n))] if fq else []


_DB = {}


def pick(query, positive, cands, k, taken, key=None):
    """Top-k of cands (texts) by BM25 against the query; not the positive, not near-copies, not already taken.
    key: cache the FTS index of a candidate list that is reused across queries."""
    if not cands or k <= 0:
        return []
    if key is None:
        db = fts(cands)
    else:
        if key not in _DB:
            _DB[key] = fts(cands)
        db = _DB[key]
    order = bm25_order(db, query, 50) or list(range(min(len(cands), 50)))
    out = []
    for i in order:
        t = cands[i]
        if t == positive or t in taken or len(t.split()) < 25 or near_copy(t, positive):
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out


def load_docs(need_paths):
    """-> (sections of the needed files by path, changelog sections by project, regular sections by project)."""
    doc_secs, changelog, regular = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in open(DOCS):
        d = json.loads(line)
        if not allowed_train_doc(d) or EXCLUDE.search(d.get("project", "")) or EXCLUDE.search(d.get("path", "")):
            continue
        is_cl = bool(CHANGELOG.search(d["path"]))
        for head, _, _, t in split_doc(d["text"], d["path"].lower()):
            rec = {"project": d["project"], "source_id": d["source_id"], "path": d["path"], "heading": head, "text": t,
                   "full": f"{head}\n{t}" if head else t}
            if is_cl:
                changelog[d["project"]].append(rec)
            else:
                regular[d["project"]].append(rec)
                if d["path"] in need_paths:
                    doc_secs[d["path"]].append(rec["full"])
    return doc_secs, changelog, regular


def changelog_questions(per_call=8, workers=12, seed=0):
    """Changelog / release-note sections as POSITIVES (23 of 150 held-out gold sections are in CHANGES files, so the
    reranker must learn to tell a relevant changelog entry from a merely similar one, not to demote changelogs)."""
    from defrost_graph.memory.text_bench import code_share
    OUT.mkdir(parents=True, exist_ok=True)
    f_s, f_q = OUT / "changelog_sections.jsonl", OUT / "changelog_questions.jsonl"
    if not f_s.exists():
        _, changelog, _ = load_docs(set())
        rng = random.Random(seed)
        picked = []
        for proj, recs in sorted(changelog.items()):
            ok = [r for r in recs if 60 <= len(r["text"].split()) <= 400 and code_share(r["text"]) < 0.5]
            rng.shuffle(ok)
            picked += ok[:CL_POS_PER_PROJECT]
        rng.shuffle(picked)
        picked = picked[:MAX_CL_POS]
        print(f"changelog positives: {len(picked)} {Counter(r['project'] for r in picked).most_common(8)}", flush=True)
        f_s.write_text("".join(json.dumps(r) + "\n" for r in picked))
    secs = [json.loads(l) for l in open(f_s)]
    done = {json.loads(l)["i"] for l in open(f_q)} if f_q.exists() else set()
    batches = [list(range(b, min(b + per_call, len(secs)))) for b in range(0, len(secs), per_call)]
    batches = [b for b in batches if not all(i in done for i in b)]
    print(f"{len(secs)} changelog sections, {len(batches)} calls to go", flush=True)

    def run(b):
        body = "\n\n".join(f"[{k}] ({secs[i]['project']}: {secs[i]['path'].rsplit('/', 1)[-1]}) {secs[i]['heading']}\n"
                             f"{secs[i]['text'][:2500]}" for k, i in enumerate(b))
        items = claude(CHANGELOG_PROMPT.format(sections=body)) or []
        return [{"i": b[it["n"]], "question": it["question"].strip()} for it in items
                if 0 <= it.get("n", -1) < len(b) and it.get("usable") and it.get("question", "").strip()]

    with ThreadPoolExecutor(workers) as ex, open(f_q, "a") as f:
        for k, rows in enumerate(ex.map(run, batches)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(batches)} calls", flush=True)


# A group before filtering: new hard negatives in "cand" (with their kind), and "fallback" negatives used to fill the
# group back to N_NEG after the false-negative filter (v1's BM25 negatives, or further BM25 candidates).
def techdoc_groups(v1_keep, doc_secs, changelog):
    secs = [json.loads(l) for l in open(SUP / "techdoc_sections.jsonl")]
    qs = [json.loads(l) for l in open(SUP / "techdoc_questions.jsonl")]
    cl_text = {p: [r["full"] for r in v] for p, v in changelog.items()}
    all_cl = [t for v in cl_text.values() for t in v]
    out, stats = [], Counter()
    for q in qs:
        s = secs[q["i"]]
        pos = f"{s['heading']}\n{s['text']}" if s["heading"] else s["text"]
        v1 = v1_keep.get((q["question"], pos))
        if v1 is None:
            continue
        sib = pick(q["question"], pos, doc_secs.get(s["path"], []), MAX_SIB + 2, set(), key=("f", s["path"]))
        cl = pick(q["question"], pos, cl_text.get(s["project"], []), MAX_CHANGELOG + 2, set(sib), key=("p", s["project"]))
        if not cl:                                      # no changelog in this project: one from any project
            cl = pick(q["question"], pos, all_cl, 1, set(sib), key="all")
        stats["sib"] += bool(sib); stats["cl"] += bool(cl); stats["n"] += 1
        out.append({"source": "techdoc", "source_id": s["source_id"], "query": q["question"], "positive": pos,
                    "cand": [[t, "sibling"] for t in sib] + [[t, "changelog"] for t in cl],
                    "fallback": [n for n in v1["negatives"] if n != pos]})
    print(f"techdoc groups {stats['n']}: with siblings {stats['sib']}, with changelog {stats['cl']}", flush=True)
    return out


def changelog_groups(changelog, regular):
    secs = [json.loads(l) for l in open(OUT / "changelog_sections.jsonl")]
    qs = [json.loads(l) for l in open(OUT / "changelog_questions.jsonl")]
    cl_text = {p: [r["full"] for r in v] for p, v in changelog.items()}
    reg_text = {p: [r["full"] for r in v] for p, v in regular.items()}
    out = []
    for q in qs:
        s = secs[q["i"]]
        pos = s["full"]
        cl = pick(q["question"], pos, cl_text.get(s["project"], []), 4, set(), key=("p", s["project"]))
        reg = pick(q["question"], pos, reg_text.get(s["project"], []), 10, set(cl), key=("r", s["project"]))
        out.append({"source": "changelog", "source_id": s["source_id"], "query": q["question"], "positive": pos,
                    "cand": [[t, "changelog"] for t in cl] + [[t, "doc"] for t in reg], "fallback": []})
    print(f"changelog-positive groups {len(out)}", flush=True)
    return out


def prose_groups():
    secs = [json.loads(l) for l in open(PROSE / "sections.jsonl")]
    qs = [json.loads(l) for l in open(PROSE / "questions.jsonl")]
    text = [f"{s['heading']}\n{s['text']}" for s in secs]
    by_book, by_ch = defaultdict(list), defaultdict(list)
    for i, s in enumerate(secs):
        by_book[s["book"]].append(i); by_ch[s["chapter"]].append(i)
    out = []
    for q in qs:
        i = q["i"]
        s = secs[i]
        if s["role"] != "train":
            continue
        sib = pick(q["question"], text[i], [text[j] for j in by_ch[s["chapter"]] if j != i], MAX_SIB + 2, set())
        rest = pick(q["question"], text[i], [text[j] for j in by_book[s["book"]]], 12, set(sib), key=("b", s["book"]))
        out.append({"source": "prose", "source_id": s["source_id"], "query": q["question"], "positive": text[i],
                    "book": s["book"],
                    "cand": [[t, "sibling"] for t in sib] + [[t, "book"] for t in rest], "fallback": []})
    print(f"prose groups {len(out)}: {Counter(o['book'] for o in out)}", flush=True)
    return out


CAPS = {"sibling": MAX_SIB, "changelog": MAX_CHANGELOG, "doc": N_NEG, "book": N_NEG}


def assemble(grp, keep):
    """keep[j]: candidate j passed the false-negative filter -> final negatives (capped per kind), filled from fallback."""
    negs, kinds, used = [], [], Counter()
    for (t, kind), ok in zip(grp["cand"], keep):
        if ok and used[kind] < CAPS[kind] and len(negs) < N_NEG:
            negs.append(t); kinds.append(kind); used[kind] += 1
    for t in grp["fallback"]:
        if len(negs) >= N_NEG:
            break
        if t not in negs:
            negs.append(t)
    return negs, kinds


def candidates():
    """Step 1 (defrost venv): all candidate groups, leakage-gated -> OUT/candidates.jsonl."""
    v1 = {s: [json.loads(l) for l in open(SUP / f"rerank_{s}.jsonl")] for s in ("train", "val")}
    v1_keep = {(g["query"], g["positive"]): g for s in v1.values() for g in s if g["source"] == "techdoc"}
    need = {json.loads(l)["path"] for l in open(SUP / "techdoc_sections.jsonl")}
    doc_secs, changelog, regular = load_docs(need)
    print(f"siblings from {len(doc_secs)} files; changelog sections {sum(map(len, changelog.values()))} in "
          f"{len(changelog)} projects", flush=True)
    groups = techdoc_groups(v1_keep, doc_secs, changelog) + changelog_groups(changelog, regular) + prose_groups()
    g = eval_grams()
    kept, dropped = [], Counter()
    for grp in groups:
        if grams13(grp["query"]) & g or grams13(grp["positive"]) & g:
            dropped[grp["source"]] += 1
            continue
        grp["cand"] = [c for c in grp["cand"] if not (grams13(c[0]) & g)]
        grp["fallback"] = [t for t in grp["fallback"] if not (grams13(t) & g)]
        kept.append(grp)
    (OUT / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (OUT / "candidates_meta.json").write_text(json.dumps({"n": len(kept), "by_source": Counter(r["source"] for r in kept),
                                                          "leakage_dropped": dropped}, indent=1))
    print(f"candidates {len(kept)} {Counter(r['source'] for r in kept)}; leakage dropped {dict(dropped)}", flush=True)


def build(seed=0, n_val_new=256):
    """Step 3 (defrost venv), after rerank_v2_filter.py wrote OUT/candidates_keep.jsonl."""
    rng = random.Random(seed)
    v1 = {s: [json.loads(l) for l in open(SUP / f"rerank_{s}.jsonl")] for s in ("train", "val")}
    split_of = {(g["query"], g["positive"]): s for s, gs in v1.items() for g in gs if g["source"] == "techdoc"}
    groups = [json.loads(l) for l in open(OUT / "candidates.jsonl")]
    keeps = [json.loads(l)["keep"] for l in open(OUT / "candidates_keep.jsonl")]
    assert len(groups) == len(keeps)
    kept, short, kinds_n, filtered = [], Counter(), Counter(), Counter()
    for grp, keep in zip(groups, keeps):
        filtered[grp["source"]] += sum(not k for k in keep)
        negs, kinds = assemble(grp, keep)
        if len(negs) < N_NEG:
            short[grp["source"]] += 1
            continue
        kinds_n.update(kinds)
        kept.append({"source": grp["source"], "source_id": grp["source_id"], "query": grp["query"],
                     "positive": grp["positive"], "negatives": negs,
                     "kinds": kinds, "instruction": TECHDOC_INSTRUCTION})
    rng.shuffle(kept)
    new_val, new_tr = [], []
    n_side = Counter()
    for x in kept:
        if x["source"] == "techdoc":
            (new_val if split_of[(x["query"], x["positive"])] == "val" else new_tr).append(x)
        elif n_side[x["source"]] < n_val_new // 4:              # a slice of prose / changelog groups for val
            n_side[x["source"]] += 1; new_val.append(x)
        else:
            new_tr.append(x)
    public = [x for x in v1["train"] if x["source"] != "techdoc"]
    rng.shuffle(public)
    replay = public[:len(new_tr)]                                 # half of every step stays v1-style public data
    train = new_tr + replay
    rng.shuffle(train)
    rng.shuffle(new_val)
    val = new_val[:n_val_new]
    written = gate_and_write("rerank_v2", {str(OUT / f"{name}.jsonl"): part for name, part in
                                           (("rerank_train", train), ("rerank_val", val), ("rerank_val_v1", v1["val"]))},
                             inputs=[OUT / "candidates.jsonl", OUT / "candidates_keep.jsonl"])
    train, val = list(written.values())[:2]
    meta = json.loads((OUT / "candidates_meta.json").read_text())
    manifest = {"n_train": len(train), "train_by_source": Counter(x["source"] for x in train),
                "n_val": len(val), "val_by_source": Counter(x["source"] for x in val),
                "new_negative_kinds": kinds_n, "false_negative_filtered": filtered, "dropped_short": short,
                "leakage_dropped": meta["leakage_dropped"],
                "negatives": {
                    "techdoc": f"<= {MAX_SIB} same-file siblings + <= {MAX_CHANGELOG} changelog/release-note sections "
                               "(same project; one from any project if the project has none), BM25-ranked; rest v1 BM25",
                    "changelog": f"positive = a changelog section; <= {MAX_CHANGELOG} other changelog sections + regular "
                                 "doc sections of the same project, BM25-ranked",
                    "prose": f"<= {MAX_SIB} same-chapter siblings, rest BM25 over the same book",
                    "public": "v1 groups unchanged (replay)",
                    "false_negative_filter": f"a new negative is dropped if its Defrost-Ret-B cosine to the query is "
                                             f">= {FILTER_MARGIN} x the positive's (NV-Retriever TopK-PercPos)"},
                "books": {k: {"license": v[2], "role": v[3]} for k, v in BOOKS.items()},
                "gate": "13-gram overlap vs every eval suite question/answer and every section of the eval KBs "
                        "(sup_data.eval_grams), applied to query, positive and each negative",
                "sha256": {n: hashlib.sha256((OUT / f"{n}.jsonl").read_bytes()).hexdigest()
                           for n in ("rerank_train", "rerank_val", "rerank_val_v1")}}
    (OUT / "MANIFEST_RERANK").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("books", "negatives", "gate")}, indent=1))


if __name__ == "__main__":
    {"prose": prose, "changelog_questions": changelog_questions, "candidates": candidates, "build": build,
     "dev_suite": dev_suite}[sys.argv[1]]()
