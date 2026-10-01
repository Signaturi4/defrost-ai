"""Tech-doc pretraining corpus for the KG-BiLM MNTP -> CGSA recipe (plan P2, docs/jev_for_graph/v2/
tech_doc_graph_training_plan.md). One "mega repository" of every technical document, then one cleaned corpus.

    uv run python -m defrost_graph.bilm.techdoc_corpus all          # gather -> hf -> synth -> clean -> corpus
    uv run python -m defrost_graph.bilm.techdoc_corpus gather       # or one stage at a time

Layout under defrost_graph/data/techdoc_mega/ (git-ignored; everything below is regenerable from this file):

  raw/local/<root>/<relpath>   every .md, plus .rst/.mdx/.txt inside a docs/ folder, from the local projects
  raw/oss/<repo>/<relpath>     the same for the OSS repos cloned for P1 (defrost_graph copy/data/techdoc/repos)
  raw/hf/<dataset>/            stackoverflow-ner, NER-RE-for-Software-Mentions, SciERC (full release), CrossRE ai
  raw/hfdocs/<name>/<id>.md    open document sets used as whole docs: PEPs, python Stack Overflow threads
  raw/hf/csn_docstrings.txt    CodeSearchNet python docstrings (train partition), one per line
  raw/synth/<repo>.md          code-derived docs (below)
  raw/manifest.jsonl           one row per raw file: source, project, path, sha256, bytes
  clean/docs.jsonl             cleaned, deduplicated, secret-scrubbed documents with a train / heldout split
  corpus/corpus_mntp.txt       MNTP input  (run_kmp.py, train_file)
  corpus/corpus_cgsa.txt       CGSA input  (run_cgsa.py, Wiki1M format: one sentence per line)
  corpus/corpus_heldout.txt    held-out text for the "MNTP accuracy up on doc text" gate
  manifest.json                sha256 of every corpus file + counts

Why each choice, and where it is grounded:

  * Domain data over generic text. The reference KMP config pretrains on wikitext; ModernBERT (Warner et al.
    2024, section 2.2.1) trains on a mixture that includes code, and reports that this is why it wins CodeSearchNet
    and StackQA without losing on text. So the corpus is docs WITH their inline code, not prose alone, and code
    blocks are kept (capped) in the MNTP input.
  * Mixture balance. One project (a third-party skills repo) has ~1.2k markdown files; left alone it would be a
    fifth of the corpus. Every project is capped at MAX_PROJECT_SHARE of the corpus characters -- the same
    concern as any-domain's p(source) ~ n^0.5 sampling (data_requirements.md D5).
  * Held-out repos never enter pretraining. defrost (this repo) and jinja / marshmallow / werkzeug are P1's dev and test
    repos. MNTP on their text would be training on the eval distribution, so their docs are written to
    corpus_heldout.txt only, and used to measure the P2 gate.
  * Secrets. Local notes and READMEs carry API keys; the corpus is uploaded to a Modal volume, so every document is
    scrubbed (key patterns, bearer tokens, JWTs, private-key blocks, long high-entropy strings, e-mail addresses)
    before anything leaves the machine. raw/ never leaves the machine.
  * Open document sets (all openly licensed; each is one project, so MAX_PROJECT_SHARE caps it):
    PEPs (common-pile, public domain) are long technical design docs with code, close to the plan / README register;
    python Stack Overflow threads (HuggingFaceTB/stackexchange_2025_md, CC BY-SA) are the "why does X call Y" register
    ModernBERT's StackQA result credits; CodeSearchNet python docstrings (Nan-Do/code-search-net-python, Apache-2.0,
    train partition only: CoIR evaluates on the CSN test partition in P4) give API prose tied to real functions.
    Threads tagged with a held-out library (jinja2 / werkzeug / marshmallow) and CSN rows from those repos are
    dropped. Not used: CoRNStack (CodeRankEmbed's training data -- P4's baseline), CoNaLa (intents under 8 words).
  * Synthetic docs are generated from code, not by an LLM: free, deterministic, and grounded -- every claim in them
    is an AST fact. Per-module API references (signature + docstring, the RepoAgent shape) and per-folder
    summaries (the CodeWiki hierarchy, built from the P0 graph's contains / inherits / calls / imports / flag / env
    edges). They teach the relation vocabulary of the schema (part_of, inherits, calls, depends_on, configures) in
    the documentation register. Capped at MAX_SYNTH_SHARE so templated text cannot dominate.
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEGA = REPO / "defrost_graph/data/techdoc_mega"
HOME = Path.home()

LOCAL_ROOTS = []                       # private local project folders (redacted in the public copy)
OSS_DIR = REPO / "defrost_graph copy/data/techdoc/repos"
OSS_TRAIN = ["attrs", "black", "click", "django", "fastapi", "flask", "httpx", "pydantic", "pytest", "requests",
             "rich", "sqlalchemy", "sqlmodel", "starlette", "typer"]
OSS_HELDOUT = ["jinja", "marshmallow", "werkzeug"]
HELDOUT_LOCAL = [str(REPO)]                          # defrost: P1's own-repo test set

SKIP_PARTS = {"node_modules", "site-packages", ".cache", "Library", "dist", "build", ".next", ".git",
              "__pycache__", ".history", ".Trash", ".pytest_cache", ".mypy_cache", "venv", "env", ".tox",
              "agent_envoirment_ml", "defrost_graph copy", "techdoc_mega"}
SKIP_NAMES = re.compile(r"^(license|licence|copying|code_of_conduct|contributors|authors|notice)(\.|$)", re.I)
PERSONAL = re.compile(r"/Documents/Main/(Personal[^/]*|Business)/")     # life notes, not technical documents
DOC_EXT = {".md", ".markdown", ".mdx", ".rst", ".txt"}

MAX_PROJECT_SHARE = 0.12
MAX_SYNTH_SHARE = 0.20
MIN_CHARS = 300
CODE_BLOCK_MAX_LINES = 40

HF_DATASETS = {"stackoverflow-ner": "mrm8488/stackoverflow-ner",
               "NER-RE-for-Software-Mentions": "psresearch/NER-RE-for-Software-Mentions"}
PEP_REPO = "common-pile/python_enhancement_proposals_filtered"
SO_REPO, SO_DIR, SO_SHARDS = "HuggingFaceTB/stackexchange_2025_md", "stackoverflow.com", 177
SO_ROW_GROUPS, SO_MIN_SCORE, SO_MAX_THREADS = 120, 2, 4000   # ~1000 rows per row group, ~7% tagged python
CSN_REPO, CSN_FILE = "Nan-Do/code-search-net-python", "data/train-00000-of-00004-ee77a7de79eb2ab2.parquet"
CSN_MAX = 40000
HELDOUT_LIBS = re.compile(r"jinja|werkzeug|marshmallow", re.I)
SCIERC_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"


# ── gather ───────────────────────────────────────────────────────────────────

def _skip(path):
    parts = set(Path(path).parts)
    if parts & SKIP_PARTS or any(p.startswith(".venv") for p in parts):
        return True
    return bool(PERSONAL.search(str(path))) or bool(SKIP_NAMES.match(Path(path).name))


def _is_doc(path):
    p = Path(path)
    if p.suffix.lower() in (".md", ".markdown"):
        return True
    return p.suffix.lower() in DOC_EXT and any(part.lower() in ("docs", "doc", "documentation") for part in p.parts)


def _candidates(root):
    """Spotlight when available (a filesystem walk of ~/Downloads and ~/Desktop takes minutes), else os.walk."""
    root = Path(root)
    try:
        query = " || ".join(f'kMDItemFSName == "*{ext}"c' for ext in sorted(DOC_EXT))
        out = subprocess.run(["mdfind", "-onlyin", str(root), query],
                             capture_output=True, text=True, timeout=300)
        if out.returncode == 0 and out.stdout.strip():
            return [Path(x) for x in out.stdout.splitlines() if x]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    found = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_PARTS and not d.startswith(".venv") and not d.startswith(".")]
        found += [Path(dp) / f for f in fns]
    return found


SF_DATALESS = 0x40000000      # macOS: an iCloud placeholder; reading it blocks on a cloud download


def _is_dataless(path):
    try:
        return bool(os.stat(path).st_flags & SF_DATALESS)
    except (OSError, AttributeError):
        return False


def _copy(src, dst, source, project, manifest):
    if _is_dataless(src):
        manifest.append({"source": source, "project": project, "origin": str(src), "skipped": "dataless"})
        return
    try:
        data = src.read_bytes()
    except OSError:
        return
    if not data or len(data) > 2_000_000:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    manifest.append({"source": source, "project": project, "path": str(dst.relative_to(MEGA)),
                     "origin": str(src), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})


def _project_of(path):
    """~/<root>/<project>/... -> '<root>/<project>'; defrost keeps its own name."""
    rel = Path(path).relative_to(HOME).parts
    return "/".join(rel[:2]) if len(rel) > 2 else rel[0]


def gather():
    manifest = []
    raw = MEGA / "raw"
    for sub in ("local", "oss"):
        shutil.rmtree(raw / sub, ignore_errors=True)
    n_local = 0
    for root in LOCAL_ROOTS:
        base = HOME / root
        if not base.exists():
            continue
        for p in _candidates(base):
            if not p.is_file() or _skip(p) or not _is_doc(p):
                continue
            rel = p.relative_to(HOME)
            _copy(p, raw / "local" / rel, "local", _project_of(p), manifest)
            n_local += 1
    for repo in OSS_TRAIN + OSS_HELDOUT:
        base = OSS_DIR / repo
        if not base.exists():
            print(f"  missing OSS repo {repo} (expected at {base})")
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in SKIP_PARTS and not d.startswith(".")]
            for f in fns:
                p = Path(dp) / f
                if _is_doc(p) and not _skip(p.relative_to(base)):
                    _copy(p, raw / "oss" / repo / p.relative_to(base), "oss", f"oss/{repo}", manifest)
    with open(raw / "manifest.jsonl", "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")
    by_src = collections.Counter(r["source"] for r in manifest)
    print(f"gather: {len(manifest)} files ({dict(by_src)}), "
          f"{sum(r.get('bytes', 0) for r in manifest) / 1e6:.1f} MB, {len({r['project'] for r in manifest})} projects")


# ── HF / public datasets ─────────────────────────────────────────────────────

def hf():
    from huggingface_hub import snapshot_download
    out = MEGA / "raw/hf"
    out.mkdir(parents=True, exist_ok=True)
    for name, repo_id in HF_DATASETS.items():
        path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=out / name)
        print(f"hf: {repo_id} -> {path}")
    scierc = out / "SciERC"
    if not (scierc / "processed_data").exists():
        scierc.mkdir(exist_ok=True)
        tgz = scierc / "sciERC_processed.tar.gz"
        try:
            urllib.request.urlretrieve(SCIERC_URL, tgz)
            with tarfile.open(tgz) as t:
                t.extractall(scierc)
            print(f"hf: SciERC full release -> {scierc}")
        except Exception as e:          # the UW mirror is the only source of the full release; report, don't fake it
            print(f"hf: SciERC download FAILED ({e}); get it from {SCIERC_URL}")
    crossre = out / "crossre_ai"
    crossre.mkdir(exist_ok=True)
    for split in ("heldout_dev", "heldout_test"):
        src = REPO / f"defrost_graph/data/any_domain/v1/{split}/crossre_ai.jsonl"
        if src.exists():
            shutil.copy2(src, crossre / f"{split}.jsonl")
    print(f"hf: CrossRE ai (from any-domain v1) -> {crossre}")
    hf_docs()


def _write_docs(name, docs):
    d = MEGA / "raw/hfdocs" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i, text in enumerate(docs):
        (d / f"{i:05d}.md").write_text(text)
    print(f"hf: {name}: {len(docs)} docs, {sum(map(len, docs)) / 1e6:.1f}M chars")


def hf_docs(seed=0):
    import gzip
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download
    rng = random.Random(seed)
    # PEPs: one dolma json.gz, ~10 MB
    peps = []
    for line in gzip.open(hf_hub_download(PEP_REPO, "peps-dolma-0000.json.gz", repo_type="dataset"), "rt"):
        peps.append(json.loads(line)["text"])
    _write_docs("peps", peps)
    # Stack Overflow: read random row groups (column projection, range requests) instead of 90 GB of shards
    fs, threads = HfFileSystem(), []
    picks = sorted({(rng.randrange(SO_SHARDS), rng.random()) for _ in range(SO_ROW_GROUPS)})
    files = {}
    for shard, u in picks:
        if len(threads) >= SO_MAX_THREADS:
            break
        if shard not in files:
            files[shard] = pq.ParquetFile(fs.open(
                f"datasets/{SO_REPO}/{SO_DIR}/train-{shard:05d}-of-{SO_SHARDS:05d}.parquet"))
        pf = files[shard]
        df = pf.read_row_group(int(u * pf.num_row_groups), columns=["Tags", "Score", "ThreadText"]).to_pandas()
        df = df[df.Tags.str.contains("<python") & (df.Score >= SO_MIN_SCORE) & ~df.Tags.str.contains(HELDOUT_LIBS)]
        threads += df.ThreadText.tolist()
    _write_docs("stackoverflow_python", threads[:SO_MAX_THREADS])
    # CodeSearchNet docstrings: first paragraph of each train-partition docstring
    df = pd.read_parquet(hf_hub_download(CSN_REPO, CSN_FILE, repo_type="dataset"),
                         columns=["repo", "docstring", "partition"])
    df = df[(df.partition == "train") & ~df.repo.str.contains(HELDOUT_LIBS)]
    out, seen = [], set()
    for doc in df.docstring:
        first = " ".join(re.split(r"\n\s*\n", doc.strip())[0].split())
        if len(first.split()) >= 8 and first.lower() not in seen:
            seen.add(first.lower())
            out.append(first)
    rng.shuffle(out)
    (MEGA / "raw/hf/csn_docstrings.txt").write_text("\n".join(out[:CSN_MAX]) + "\n")
    print(f"hf: csn_docstrings: {min(len(out), CSN_MAX)} of {len(out)} docstrings")


def hf_train_sentences():
    """Plain sentences from the TRAIN splits of the public sets, for the pretraining corpus. Their dev/test splits
    stay out: P1 evaluates on them."""
    base = MEGA / "raw/hf"
    sents = []
    for name in HF_DATASETS:
        for p in sorted((base / name).rglob("*")):
            if "test" in p.name.lower() or "dev" in p.name.lower() or "valid" in p.name.lower():
                continue
            if ".cache" in p.parts:                 # huggingface_hub download metadata, not data
                continue
            if p.name == "train_texts.txt":         # NER-RE-for-Software-Mentions: one sentence per line
                sents += [l.strip() for l in open(p) if l.strip()]
            elif p.suffix == ".parquet":
                import pandas as pd
                df = pd.read_parquet(p)
                col = next((c for c in ("tokens", "words", "text", "sentence") if c in df.columns), None)
                if col:
                    sents += [" ".join(x) if not isinstance(x, str) else x for x in df[col]]
            elif p.suffix in (".json", ".jsonl"):
                with open(p) as fh:
                    head = fh.read(1)
                    fh.seek(0)
                    rows = json.load(fh) if head == "[" else [json.loads(l) for l in fh if l.strip()]
                for r in rows if isinstance(rows, list) else []:
                    if isinstance(r, dict):
                        x = r.get("tokens") or r.get("words") or r.get("text") or r.get("sentence")
                        if x:
                            sents.append(" ".join(x) if isinstance(x, list) else x)
    csn = base / "csn_docstrings.txt"
    if csn.exists():
        sents += [l.strip() for l in open(csn) if l.strip()]
    scierc = base / "SciERC/processed_data/json/train.json"
    if scierc.exists():
        for line in open(scierc):
            r = json.loads(line)
            sents += [" ".join(s) for s in r["sentences"]]
    return [s for s in sents if len(s.split()) >= 5]


# ── synthetic, code-derived docs ─────────────────────────────────────────────

def _sig(fn):
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args] + ([f"*{a.vararg.arg}"] if a.vararg else []) \
        + [x.arg for x in a.kwonlyargs] + ([f"**{a.kwarg.arg}"] if a.kwarg else [])
    return f"{fn.name}({', '.join(names)})"


def _first_para(doc):
    return (doc or "").strip().split("\n\n")[0].strip()


def _api_reference(repo_root, relpath):
    """RepoAgent-style reference for one module: signature + first docstring paragraph per public symbol."""
    try:
        tree = ast.parse((repo_root / relpath).read_text(errors="replace"))
    except (SyntaxError, ValueError):
        return ""
    mod = relpath[:-3].replace("/", ".").removesuffix(".__init__")
    out = [f"# Module `{mod}`", ""]
    if ast.get_docstring(tree):
        out += [_first_para(ast.get_docstring(tree)), ""]
    n = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            d = _first_para(ast.get_docstring(node))
            out += [f"## `{mod}.{_sig(node)}`", ""] + ([d, ""] if d else [])
            n += 1
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            head = f"## class `{mod}.{node.name}`" + (f", a subclass of `{'`, `'.join(bases)}`" if bases else "")
            out += [head, ""]
            if ast.get_docstring(node):
                out += [_first_para(ast.get_docstring(node)), ""]
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                    d = _first_para(ast.get_docstring(m))
                    out.append(f"- `{node.name}.{_sig(m)}`" + (f": {d.splitlines()[0]}" if d else ""))
            out.append("")
            n += 1
    return "\n".join(out) if n else ""


_FOLDER_TEMPLATES = [
    "The `{d}` package contains {n} modules: {mods}.",
    "`{d}` is made of {n} modules ({mods}).",
    "Inside `{d}` there are {n} modules: {mods}.",
]
_REL_TEMPLATES = {
    "inherits": ["`{a}` inherits from `{b}`.", "`{a}` is a subclass of `{b}`.", "`{a}` extends `{b}`."],
    "calls": ["`{a}` calls `{b}`.", "`{a}` uses `{b}`.", "Internally, `{a}` relies on `{b}`."],
    "imports": ["`{a}` imports `{b}`.", "`{a}` depends on `{b}`."],
    "defines_flag": ["The `{b}` option is defined in `{a}`.", "`{a}` accepts the `{b}` command-line flag."],
    "uses_env": ["`{a}` reads the `{b}` environment variable.", "`{a}` is configured by `{b}`."],
}


def _short(node_id):
    return node_id.split("::")[-1] if "::" in node_id else node_id.removeprefix("env:")


def _folder_summaries(graph, rng):
    """CodeWiki-style hierarchy: one paragraph per folder, facts only from the P0 graph's edges."""
    nodes = graph.nodes
    children = collections.defaultdict(list)
    out_edges = collections.defaultdict(list)
    for s, d, k in graph.edges:
        if k == "contains":
            children[s].append(d)
        else:
            out_edges[s].append((d, k))
    paras = []
    for did, n in sorted(nodes.items()):
        if n.kind != "dir":
            continue
        files = [c for c in children[did] if nodes[c].kind == "file" and c.endswith(".py")]
        if not files:
            continue
        d = did.removeprefix("dir:")
        mods = ", ".join(f"`{Path(f).name}`" for f in files[:12])
        lines = [f"## `{d}`", "", rng.choice(_FOLDER_TEMPLATES).format(d=d, n=len(files), mods=mods)]
        for f in files[:12]:
            syms = [c for c in children[f] if nodes[c].kind in ("class", "function")]
            if syms:
                lines.append(f"`{Path(f).name}` defines " + ", ".join(f"`{_short(s)}`" for s in syms[:8]) + ".")
            facts = []
            for sym in [f] + syms:
                for dst, k in out_edges.get(sym, []):
                    if k in _REL_TEMPLATES and dst in nodes:
                        facts.append(rng.choice(_REL_TEMPLATES[k]).format(a=_short(sym), b=_short(dst)))
                for c in children.get(sym, []):
                    for dst, k in out_edges.get(c, []):
                        if k in _REL_TEMPLATES and dst in nodes:
                            facts.append(rng.choice(_REL_TEMPLATES[k]).format(a=_short(c), b=_short(dst)))
            rng.shuffle(facts)
            lines += facts[:10]
        paras.append("\n".join(lines))
    return "\n\n".join(paras)


def synth(seed=0):
    from defrost_graph.techdoc.graph import build
    rng = random.Random(seed)
    out = MEGA / "raw/synth"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for repo in OSS_TRAIN:
        root = OSS_DIR / repo
        if not root.exists():
            continue
        g = build(root, exclude=("tests", "test", "docs", "examples", "benchmarks"))
        refs = [_api_reference(root, n.path) for n in g.nodes.values()
                if n.kind == "file" and n.path.endswith(".py") and not n.error]
        text = "\n\n".join([r for r in refs if r] + [_folder_summaries(g, rng)])
        (out / f"{repo}.md").write_text(text)
        print(f"synth: {repo}: {sum(bool(r) for r in refs)} module references, {len(text) / 1e3:.0f} kB")


# ── clean ────────────────────────────────────────────────────────────────────

SECRET_PATTERNS = [
    (re.compile(r"(?:/Users|/home)/[A-Za-z0-9._-]+"), "~"),       # home paths name the user and their layout
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "<PRIVATE_KEY>"),
    (re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{16,}"), "<SECRET>"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"), "<SECRET>"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), "<SECRET>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<SECRET>"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "<SECRET>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "<SECRET>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"), r"\1<SECRET>"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|auth)(\s*[:=]\s*[\"']?)[^\s\"'`]{8,}"),
     r"\1\2<SECRET>"),
    (re.compile(r"\b[A-Za-z0-9+/_\-]{40,}={0,2}(?![\w/])"), "<SECRET>"),     # long high-entropy blobs
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
]
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
_FENCE = re.compile(r"^(```|~~~)[^\n]*\n(.*?)^\1[ \t]*$", re.S | re.M)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG = re.compile(r"</?[A-Za-z][^>]{0,200}>")
_RST_DIRECTIVE = re.compile(r"^\.\. [\w:-]+::.*$", re.M)
_TABLE_RULE = re.compile(r"^[\s|:+=\-]{3,}$", re.M)


def scrub(text):
    for pat, rep in SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text


def _cap_block(m):
    lines = m.group(2).splitlines()
    body = "\n".join(lines[:CODE_BLOCK_MAX_LINES])
    return f"{m.group(1)}\n{body}\n{m.group(1)}"


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = _FRONT.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    text = _FENCE.sub(_cap_block, text)
    text = _IMG.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("URL", text)
    text = _TAG.sub("", text)
    text = _RST_DIRECTIVE.sub("", text)
    text = _TABLE_RULE.sub("", text)
    text = scrub(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _latin_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(len(letters), 1)


def _paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def clean():
    rows = [r for r in map(json.loads, open(MEGA / "raw/manifest.jsonl")) if "skipped" not in r]
    for p in sorted((MEGA / "raw/synth").glob("*.md")):
        rows.append({"source": "synth", "project": f"synth/{p.stem}", "path": str(p.relative_to(MEGA))})
    for p in sorted((MEGA / "raw/hfdocs").glob("*/*.md")):
        rows.append({"source": "hf", "project": f"hf/{p.parent.name}", "path": str(p.relative_to(MEGA))})
    heldout_projects = {f"oss/{r}" for r in OSS_HELDOUT} | {"research/defrost"}
    seen_para, docs, dropped = set(), [], collections.Counter()
    # heldout first, so a paragraph shared with a train doc is removed from TRAIN, never from the eval text
    rows.sort(key=lambda r: (r["project"] not in heldout_projects, r["source"] == "synth", r["path"]))
    for r in rows:
        raw = (MEGA / r["path"]).read_bytes().decode("utf-8", errors="replace")
        if raw.count("�") > 0.01 * max(len(raw), 1):
            dropped["binary"] += 1
            continue
        text = clean_text(raw)
        if _latin_share(text) < 0.7:
            dropped["non_latin"] += 1
            continue
        keep = []
        for para in _paragraphs(text):
            key = re.sub(r"\W+", " ", para.lower()).strip()
            if len(key) >= 40:
                if key in seen_para:
                    continue
                seen_para.add(key)
            keep.append(para)
        text = "\n\n".join(keep)
        if len(text) < MIN_CHARS:
            dropped["short_or_duplicate"] += 1
            continue
        split = "heldout" if r["project"] in heldout_projects else "train"
        docs.append({"id": hashlib.sha1(r["path"].encode()).hexdigest()[:12], "source": r["source"],
                     "project": r["project"], "path": r["path"], "split": split, "text": text,
                     "n_chars": len(text)})
    (MEGA / "clean").mkdir(exist_ok=True)
    with open(MEGA / "clean/docs.jsonl", "w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    c = collections.Counter((d["split"], d["source"]) for d in docs)
    print(f"clean: kept {len(docs)} docs {dict(c)}; dropped {dict(dropped)}")


# ── corpus ───────────────────────────────────────────────────────────────────

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(\"'])")


def _balance(docs, rng):
    """Cap each project at MAX_PROJECT_SHARE and synthetic text at MAX_SYNTH_SHARE of total characters."""
    total = sum(d["n_chars"] for d in docs)
    by_proj = collections.defaultdict(list)
    for d in docs:
        by_proj[d["project"]].append(d)
    kept = []
    for proj, ds in by_proj.items():
        rng.shuffle(ds)
        budget, used = MAX_PROJECT_SHARE * total, 0
        for d in ds:
            if used + d["n_chars"] > budget and used > 0:
                continue
            kept.append(d)
            used += d["n_chars"]
    real = sum(d["n_chars"] for d in kept if d["source"] != "synth")
    synth_budget = MAX_SYNTH_SHARE / (1 - MAX_SYNTH_SHARE) * real
    out, used = [], 0
    for d in kept:
        if d["source"] == "synth":
            if used >= synth_budget:
                continue
            used += d["n_chars"]
        out.append(d)
    rng.shuffle(out)
    return out


def _prose_sentences(text):
    prose = _FENCE.sub("", text)
    for para in _paragraphs(prose):
        if para.startswith(("#", "|", "-", "*", ">", "    ")) and len(para.split()) < 12:
            continue
        for s in _SENT.split(" ".join(para.split())):
            w = s.split()
            if 8 <= len(w) <= 64 and sum(c.isalpha() for c in s) > 0.5 * len(s):
                yield s


def corpus(seed=0):
    rng = random.Random(seed)
    docs = [json.loads(l) for l in open(MEGA / "clean/docs.jsonl")]
    train = _balance([d for d in docs if d["split"] == "train"], rng)
    heldout = [d for d in docs if d["split"] == "heldout"]
    hf_sents = [scrub(s) for s in hf_train_sentences()]
    out = MEGA / "corpus"
    out.mkdir(exist_ok=True)
    # MNTP: one paragraph per line, documents contiguous; run_kmp.py concatenates lines and chunks by length
    with open(out / "corpus_mntp.txt", "w") as fh:
        for d in train:
            for para in _paragraphs(d["text"]):
                fh.write(" ".join(para.split()) + "\n")
        for s in hf_sents:
            fh.write(s + "\n")
    seen, n_cgsa = set(), 0
    with open(out / "corpus_cgsa.txt", "w") as fh:
        pool = [s for d in train for s in _prose_sentences(d["text"])] + hf_sents
        rng.shuffle(pool)
        for s in pool:
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            fh.write(s + "\n")
            n_cgsa += 1
    with open(out / "corpus_heldout.txt", "w") as fh:
        for d in heldout:
            for para in _paragraphs(d["text"]):
                if len(para) >= 120:
                    fh.write(" ".join(para.split()) + "\n")
    files = {}
    for p in sorted(out.glob("*.txt")):
        files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size,
                         "lines": sum(1 for _ in open(p))}
    mix = collections.Counter()
    for d in train:
        mix[d["source"]] += d["n_chars"]
    top = collections.Counter()
    for d in train:
        top[d["project"]] += d["n_chars"]
    total = sum(mix.values())
    manifest = {"files": files, "n_train_docs": len(train), "n_heldout_docs": len(heldout),
                "n_hf_sentences": len(hf_sents), "n_cgsa_sentences": n_cgsa,
                "train_chars": total, "mix_share": {k: round(v / total, 4) for k, v in mix.items()},
                "top_projects_share": {k: round(v / total, 4) for k, v in top.most_common(15)},
                "heldout_projects": sorted({d["project"] for d in heldout}),
                "caps": {"project": MAX_PROJECT_SHARE, "synth": MAX_SYNTH_SHARE}}
    json.dump(manifest, open(MEGA / "manifest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "top_projects_share"}, indent=1))
    print("top projects:", manifest["top_projects_share"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gather", "hf", "synth", "clean", "corpus", "all"])
    a = ap.parse_args(argv)
    MEGA.mkdir(parents=True, exist_ok=True)
    stages = {"gather": gather, "hf": hf, "synth": synth, "clean": clean, "corpus": corpus}
    for name in (stages if a.stage == "all" else [a.stage]):
        print(f"=== {name} ===", flush=True)
        stages[name]()


if __name__ == "__main__":
    main()
