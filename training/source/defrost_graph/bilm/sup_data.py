"""P2b S1: training triplets (query, positive, hard negative) for the supervised retrieval stage.

    python -m defrost_graph.bilm.sup_data synth     # tech-doc questions from the train split of techdoc_mega (resumable)
    python -m defrost_graph.bilm.sup_data public    # public E5-style sources from the HF hub
    python -m defrost_graph.bilm.sup_data build     # hard negatives, leakage gate, mix -> defrost_graph/data/sup_v1/

Plan and rationale: docs/jev_for_graph/v2/supervised_retrieval_plan.md."""
import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from defrost_graph.bilm import data_provenance as dp

DOCS = Path(os.environ.get("TECHDOC_MEGA") or "defrost_graph/data/techdoc_mega") / "clean/docs.jsonl"
OUT = Path(os.environ.get("SUP_OUT") or "defrost_graph/data/sup_v1")
EXCLUDE = re.compile(r"client|ii-|jinja|werkzeug|marshmallow|research", re.I)
SKIP_PATH = re.compile(r"(CHANGES|CHANGELOG|HISTORY|RELEASE|AUTHORS|LICENSE|CONTRIBUTORS)", re.I)
TECHDOC_INSTRUCTION = "Given a developer question about a software project, retrieve the documentation passage that answers it"
INSTRUCTIONS = {
    "msmarco": "Given a web search query, retrieve relevant passages that answer the query",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "stackexchange": "Given a question title from a technical Q&A site, retrieve the body of that question",
    "allnli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "quora": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "techdoc": TECHDOC_INSTRUCTION,
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["n", "usable", "question"],
              "properties": {"n": {"type": "integer"}, "usable": {"type": "boolean"}, "question": {"type": "string"}}}}}}
PROMPT = """You write training questions for a documentation search engine used by developers.
For EACH numbered documentation section below, write ONE question a developer would realistically ask whose answer
is in that section. Paraphrase: do not copy the heading or distinctive multi-word phrases (ordinary identifiers are
fine). The question must be specific to that section. Set usable=false for sections with no real content (tables of
contents, link lists, boilerplate).

{sections}"""


_ALLOW = None


def allowlist():
    """Source allowlist + private markers (data_provenance); strict: the private markers file must exist."""
    global _ALLOW
    if _ALLOW is None:
        _ALLOW = dp.Allowlist.load()
    return _ALLOW


def allowed_train_doc(d):
    """A clean/docs.jsonl row may feed supervised data only if it is a train-split document of an allowlisted source
    whose path passes the private-path guard. Rows from before the allowlist (no source_id) stop the build."""
    if d.get("split") != "train":
        return False
    if "source_id" not in d:
        raise dp.SourceRejected(f"{DOCS} predates the allowlist (no source_id): rebuild it with techdoc_corpus")
    allowlist().source(d["source_id"])
    allowlist().check_path(d.get("path", ""))
    return True


def gate_and_write(stage, parts, inputs=(), drop_eval=True):
    """parts: {file: rows} with rows carrying source_id. One 13-gram gate over all rows (private overlap stops the
    build; eval overlap drops the row), then each file is written with its <file>.provenance.json."""
    a = allowlist()
    flat = [(name, r) for name, rows in parts.items() for r in rows]
    units = [(r["source_id"], "\n".join(str(t) for t in [r.get("query"), r.get("positive"), r.get("negative")]
                                        + list(r.get("negatives") or []) if t)) for _, r in flat]
    res = dp.run_gate(stage, units, a, drop_eval=drop_eval)
    keep = set(res["kept"])
    out = {}
    for name in parts:
        idx = [i for i, (n, _) in enumerate(flat) if n == name and i in keep]
        rows = [flat[i][1] for i in idx]
        Path(name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        dp.write_manifest(f"{stage}:{Path(name).stem}", Path(name), dp.count_units(units, idx), a, res,
                          inputs=[p for p in inputs if Path(p).exists()])
        out[name] = rows
    return out


def shingles(text, k=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def grams13(text):
    w = re.findall(r"\w+", text.lower())
    return {hash(" ".join(w[i:i + 13])) for i in range(len(w) - 12)}


def techdoc_sections():
    from defrost_graph.memory.text_bench import code_share
    from defrost_graph.memory.text_kb import sections_of
    out = []
    for line in open(DOCS):
        d = json.loads(line)
        if not allowed_train_doc(d) or EXCLUDE.search(d.get("project", "")) or EXCLUDE.search(d.get("path", "")) \
                or SKIP_PATH.search(d.get("path", "")):
            continue
        for _, hp, _, _, text in sections_of(d["text"], rst=d["path"].lower().endswith(".rst")):
            head = " > ".join(hp)
            nw = len(text.split())
            if 60 <= nw <= 400 and code_share(text) < 0.5:
                out.append({"project": d["project"], "source_id": d["source_id"], "path": d["path"],
                            "heading": head, "text": text})
    return out


def claude(prompt):
    cmd = ["claude", "-p", prompt, "--model", "haiku", "--output-format", "json", "--tools", "", "--json-schema",
           json.dumps(SCHEMA), "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    for _ in range(3):
        try:
            out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout)
            if isinstance(out, list):
                out = next(e for e in reversed(out) if e.get("type") == "result")
            return (out.get("structured_output") or json.loads(out["result"]))["items"]
        except Exception:
            continue
    return None


def synth(n_target=27000, per_call=8, workers=14, seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    secs = techdoc_sections()
    rng = random.Random(seed)
    rng.shuffle(secs)
    # cap per project so a few big doc sets do not dominate
    per_proj, picked = Counter(), []
    for s in secs:
        if per_proj[s["project"]] < 1500:
            per_proj[s["project"]] += 1
            picked.append(s)
    picked = picked[:n_target]
    print(f"{len(secs)} eligible sections, {len(picked)} picked from {len(per_proj)} projects", flush=True)
    (OUT / "techdoc_sections.jsonl").write_text("".join(json.dumps(s) + "\n" for s in picked))
    out_f = OUT / "techdoc_questions.jsonl"
    done = {json.loads(l)["i"] for l in open(out_f)} if out_f.exists() else set()
    batches = [list(range(b, min(b + per_call, len(picked)))) for b in range(0, len(picked), per_call)]
    batches = [b for b in batches if not all(i in done for i in b)]
    print(f"{len(batches)} calls to go", flush=True)

    def run(b):
        body = "\n\n".join(f"[{k}] ({picked[i]['path']}) {picked[i]['heading']}\n{picked[i]['text'][:2500]}"
                           for k, i in enumerate(b))
        items = claude(PROMPT.format(sections=body)) or []
        return [{"i": b[it["n"]], "question": it["question"].strip()} for it in items
                if 0 <= it.get("n", -1) < len(b) and it.get("usable") and it.get("question", "").strip()]

    with ThreadPoolExecutor(workers) as ex, open(out_f, "a") as f:
        for k, rows in enumerate(ex.map(run, batches)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"  {k}/{len(batches)} calls", flush=True)


IDENT_PROMPT = """You write training questions for a documentation search engine used by developers.
For EACH numbered documentation section below, write ONE question whose answer hinges on an EXACT detail stated in
that section: a specific identifier (function, class, config key, env var, CLI flag, file path, endpoint, table or
service name) or a specific value (number, default, limit, version, port). Put that identifier or value IN the
question verbatim, the way a developer who half-remembers it would ask. Do not copy the heading. The question must be
specific to that section. Set usable=false if the section has no such concrete detail.

{sections}"""


def synth_v2(per_call=8, workers=24, ident_cap=10000):
    """E1 data: every eligible section gets (a) a normal question (reusing v1 ones) and (b) an identifier question."""
    secs = techdoc_sections()
    key = lambda s: hashlib.sha1((s["path"] + "\x00" + s["text"]).encode()).hexdigest()
    v1s = [json.loads(l) for l in open(OUT / "techdoc_sections.jsonl")]
    v1q = defaultdict(list)
    for l in open(OUT / "techdoc_questions.jsonl"):
        q = json.loads(l)
        v1q[key(v1s[q["i"]])].append(q["question"])
    per_proj, picked = Counter(), []
    for s in secs:                                   # same 1500/project cap as v1, deterministic order
        if per_proj[s["project"]] < 1500:
            per_proj[s["project"]] += 1
            picked.append(s)
    (OUT / "techdoc_sections_v2.jsonl").write_text("".join(json.dumps(s) + "\n" for s in picked))
    out_f = OUT / "techdoc_questions_v2.jsonl"
    if not out_f.exists():
        with open(out_f, "w") as f:
            for i, s in enumerate(picked):
                for q in v1q.get(key(s), []):
                    f.write(json.dumps({"i": i, "kind": "normal", "question": q}) + "\n")
    done = defaultdict(set)
    for l in open(out_f):
        r = json.loads(l)
        done[r["kind"]].add(r["i"])
    jobs = [("normal", PROMPT, [i for i in range(len(picked)) if i not in done["normal"] and key(picked[i]) not in v1q]),
            ("ident", IDENT_PROMPT, [i for i in sorted(random.Random(1).sample(range(len(picked)),
                                                                              min(ident_cap, len(picked))))
                                     if i not in done["ident"]])]
    for kind, prompt, todo in jobs:
        batches = [todo[b:b + per_call] for b in range(0, len(todo), per_call)]
        print(f"{kind}: {len(todo)} sections, {len(batches)} calls", flush=True)

        def run(b, prompt=prompt, kind=kind):
            body = "\n\n".join(f"[{k}] ({picked[i]['path']}) {picked[i]['heading']}\n{picked[i]['text'][:2500]}"
                                for k, i in enumerate(b))
            items = claude(prompt.format(sections=body)) or []
            return [{"i": b[it["n"]], "kind": kind, "question": it["question"].strip()} for it in items
                    if 0 <= it.get("n", -1) < len(b) and it.get("usable") and it.get("question", "").strip()]

        with ThreadPoolExecutor(workers) as ex, open(out_f, "a") as f:
            for k, rows in enumerate(ex.map(run, batches)):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                if k % 200 == 0:
                    print(f"  {kind} {k}/{len(batches)}", flush=True)


PUBLIC = {   # source -> (repo, file in the repo, n rows to keep)
    "msmarco": ("sentence-transformers/msmarco-bm25", "triplet/train-00000-of-00001.parquet", 18000),
    "nq": ("sentence-transformers/natural-questions", "pair/train-00000-of-00001.parquet", 7000),
    "hotpotqa": ("sentence-transformers/hotpotqa", "triplet/train-00000-of-00001.parquet", 7000),
    "allnli": ("sentence-transformers/all-nli", "triplet/train-00000-of-00001.parquet", 2500),
    "quora": ("sentence-transformers/quora-duplicates", "triplet/train-00000-of-00001.parquet", 2500),
}
SE_SITES = ["unix.stackexchange.com", "softwareengineering.stackexchange.com", "dba.stackexchange.com",
            "security.stackexchange.com", "webmasters.stackexchange.com"]


def reuse(old="defrost_graph/data/sup_v1", version=""):
    """Carry generated questions over to a rebuilt (allowlisted) corpus without new Claude calls: a question is kept
    when its section (path, heading, text) is unchanged in the new train split. -> OUT/techdoc_sections{version}.jsonl
    (every eligible section of the new corpus) + OUT/techdoc_questions{version}.jsonl (re-indexed)."""
    secs = techdoc_sections()
    idx = {(x["path"], x["heading"], x["text"]): i for i, x in enumerate(secs)}
    old = Path(old)
    old_secs = [json.loads(l) for l in open(old / f"techdoc_sections{version}.jsonl")]
    kept, by = [], Counter()
    for line in open(old / f"techdoc_questions{version}.jsonl"):
        q = json.loads(line)
        x = old_secs[q["i"]]
        j = idx.get((x["path"], x["heading"], x["text"]))
        by["kept" if j is not None else "dropped"] += 1
        if j is not None:
            kept.append({**q, "i": j})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"techdoc_sections{version}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in secs))
    (OUT / f"techdoc_questions{version}.jsonl").write_text("".join(json.dumps(q) + "\n" for q in kept))
    print(f"reuse{version}: {len(secs)} sections, questions {dict(by)}, "
          f"{len({q['i'] for q in kept})} sections with a question", flush=True)


def public(seed=0):
    """-> OUT/public_raw.jsonl rows {source, query, positive, negative|None}"""
    import gzip
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, (repo, fname, n) in PUBLIC.items():
        s = allowlist().source(src)
        if s.get("hf_id") != repo:
            raise dp.SourceRejected(f"{src}: {repo} is not the allowlisted dataset {s.get('hf_id')}")
        rev = s["revision"]
        files = [f for f in list_repo_files(repo, repo_type="dataset", revision=rev)
                 if f.startswith(fname.split("/")[0] + "/") and f.endswith(".parquet")]
        df = pd.read_parquet(hf_hub_download(repo, sorted(files)[0], repo_type="dataset", revision=rev))
        df = df.sample(n=min(n, len(df)), random_state=seed)
        cols = list(df.columns)
        for r in df.itertuples(index=False):
            r = dict(zip(cols, r))
            q = r.get("query") or r.get("anchor") or r.get("question")
            pos = r.get("positive") or r.get("answer")
            rows.append({"source": src, "source_id": s["id"], "query": q, "positive": pos, "negative": r.get("negative")})
        print(f"{src}: {n} rows ({cols})", flush=True)
    rng = random.Random(seed)
    se = []
    se_src = allowlist().source("stackexchange")
    for site in SE_SITES:
        path = hf_hub_download(se_src["hf_id"], f"{site}.jsonl.gz", repo_type="dataset", revision=se_src["revision"])
        with gzip.open(path, "rt") as f:
            for line in f:
                t = json.loads(line)["texts"]
                if len(t) >= 2 and 20 <= len(t[1].split()) <= 350:
                    se.append({"source": "stackexchange", "source_id": se_src["id"], "query": t[0], "positive": t[1],
                               "negative": None})
    rng.shuffle(se)
    rows += se[:11000]
    print(f"stackexchange: {min(11000, len(se))} of {len(se)}", flush=True)
    (OUT / "public_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def eval_grams():
    """13-grams of everything any evaluation reads: bench questions + every section of the eval text KBs,
    CodeRAG-Bench ODEX / DS-1000 queries and their gold docs."""
    import pandas as pd
    g = set()
    for bench in ["defrost_graph/memory/evals/text_heldout_v1.jsonl", "defrost_graph/memory/evals/text_client_v1.jsonl",
                  "defrost_graph/memory/evals/client_diag_v1.jsonl", "defrost_graph/memory/evals/books_curated_v1.jsonl"]:
        for line in open(bench):
            it = json.loads(line)
            g |= grams13(it["question"]) | grams13(it.get("answer", ""))
    for kb in ["~/heldout-memory/text_kb.sqlite", "~/client-memory/text_kb.sqlite", "~/books-memory/text_kb.sqlite"]:
        for (t,) in sqlite3.connect(Path(kb).expanduser()).execute("SELECT text FROM sections"):
            g |= grams13(t)
    D = Path("defrost_graph/data/public_ir")
    corpus = dict(pd.read_parquet(D / "library-documentation.parquet")[["doc_id", "doc_content"]].values)
    for name, col in [("odex", "intent"), ("ds1000", "prompt")]:
        df = pd.read_parquet(D / f"{name}.parquet")
        for q, docs in zip(df[col], df["docs"]):
            g |= grams13(q)
            for d in docs:
                g |= grams13(corpus.get(d["title"], "")[:20000])
    return g


def mine_negative(rng, bm25_db, texts, i, query, positive, same=None, k=10):
    """BM25 top-k over the source's own positives, excluding the positive and near-copies of it."""
    from defrost_graph.memory.retrieve import fts_query
    fq = fts_query(query)
    if not fq:
        return None
    ps = shingles(positive)
    cands = []
    for (rid,) in bm25_db.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?", (fq, k + 5)):
        if rid == i or (same and not same(rid)):
            continue
        sj = shingles(texts[rid])
        if len(ps & sj) / max(1, len(ps | sj)) > 0.5:
            continue
        cands.append(texts[rid])
    return rng.choice(cands[:k]) if cands else None


def fts(texts):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')")
    db.executemany("INSERT INTO t(rowid, x) VALUES (?, ?)", list(enumerate(texts)))
    return db


def build(seed=0, version=""):
    """version "" = P2b set (train/val.jsonl); "_v2" = E1 set (normal + identifier questions over all sections)."""
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(OUT / "public_raw.jsonl")]
    secs = [json.loads(l) for l in open(OUT / f"techdoc_sections{version}.jsonl")]
    qs = [json.loads(l) for l in open(OUT / f"techdoc_questions{version}.jsonl")]
    if any("source_id" not in s for s in secs):
        raise dp.SourceRejected("techdoc sections predate the allowlist (no source_id): re-run synth on a rebuilt corpus")
    for r in rows:
        r.setdefault("source_id", allowlist().source(r["source"])["id"])
    stext = [f"{s['heading']}\n{s['text']}" if s["heading"] else s["text"] for s in secs]
    sdb = fts(stext)
    for q in qs:
        i = q["i"]
        neg = mine_negative(rng, sdb, stext, i, q["question"], stext[i],
                            same=lambda r, p=secs[i]["project"]: secs[r]["project"] == p)
        if neg is None:
            neg = mine_negative(rng, sdb, stext, i, q["question"], stext[i])
        rows.append({"source": "techdoc", "source_id": secs[i]["source_id"], "query": q["question"],
                     "positive": stext[i], "negative": neg, "project": secs[i]["project"],
                     "kind": q.get("kind", "normal")})
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src, rs in by_src.items():                   # mine negatives where the source gave none
        if all(r["negative"] for r in rs):
            continue
        pos = [r["positive"] for r in rs]
        db = fts(pos)
        for i, r in enumerate(rs):
            if not r["negative"]:
                r["negative"] = mine_negative(rng, db, pos, i, r["query"], r["positive"])
    g = eval_grams()
    kept, dropped, noneg = [], Counter(), Counter()
    for r in rows:
        if not r["negative"]:
            noneg[r["source"]] += 1
            continue
        if grams13(r["query"]) & g or grams13(r["positive"]) & g or grams13(r["negative"]) & g:
            dropped[r["source"]] += 1
            continue
        r["instruction"] = INSTRUCTIONS[r["source"]]
        kept.append(r)
    rng.shuffle(kept)
    n_val = 512
    val, train = kept[:n_val], kept[n_val:]
    written = gate_and_write(f"retb_sup{version}", {str(OUT / f"train{version}.jsonl"): train,
                                                     str(OUT / f"val{version}.jsonl"): val},
                             inputs=[DOCS, OUT / "public_raw.jsonl", OUT / f"techdoc_questions{version}.jsonl"])
    train, val = written.values()
    manifest = {"n_train": len(train), "n_val": len(val), "by_source": Counter(r["source"] for r in train),
                "leakage_dropped": dropped, "no_negative_dropped": noneg,
                "gate": "13-gram overlap vs held-out/client bench questions+answers, every section of both eval KBs, "
                        "ODEX/DS-1000 queries and gold docs",
                "excluded_projects_regex": EXCLUDE.pattern, "instructions": INSTRUCTIONS,
                "sha256": {n: hashlib.sha256((OUT / f"{n}{version}.jsonl").read_bytes()).hexdigest() for n in ("train", "val")}}
    (OUT / f"manifest{version}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "instructions"}, indent=1))


def rerank(n_neg=7, seed=0):
    """Groups for the cross-encoder: each train/val row -> {query, positive, negatives[n_neg]}: the row's own hard
    negative + BM25 top-k over the same source's positives (near-copies of the positive removed). Same leakage-gated
    rows as the bi-encoder set, so the gate carries over."""
    rng = random.Random(seed)
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(OUT / f"{split}.jsonl")]
        by = defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        out = []
        for src, rs in by.items():
            pos = [r["positive"] for r in rs]
            db = fts(pos)
            from defrost_graph.memory.retrieve import fts_query
            for i, r in enumerate(rs):
                negs = [r["negative"]]
                ps = shingles(r["positive"])
                fq = fts_query(r["query"])
                if fq:
                    for (rid,) in db.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT 30", (fq,)):
                        if rid == i or pos[rid] in negs:
                            continue
                        sj = shingles(pos[rid])
                        if len(ps & sj) / max(1, len(ps | sj)) > 0.5:
                            continue
                        negs.append(pos[rid])
                        if len(negs) >= n_neg:
                            break
                while len(negs) < n_neg:                     # pad with random same-source passages
                    c = rng.choice(pos)
                    if c != r["positive"] and c not in negs:
                        negs.append(c)
                out.append({"source": src, "source_id": r["source_id"], "query": r["query"],
                            "positive": r["positive"], "negatives": negs})
        rng.shuffle(out)
        out = gate_and_write(f"rerank_v1_{split}", {str(OUT / f"rerank_{split}.jsonl"): out},
                             inputs=[OUT / f"{split}.jsonl"])[str(OUT / f"rerank_{split}.jsonl")]
        print(split, len(out), Counter(o["source"] for o in out))


if __name__ == "__main__":
    {"synth": synth, "synth_v2": synth_v2, "public": public, "build": build, "reuse": reuse,
     "reuse_v2": lambda: reuse(version="_v2"),
     "build_v2": lambda: build(version="_v2"), "rerank": rerank}[sys.argv[1]]()
