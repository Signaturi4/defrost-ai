"""Training-data provenance: only listed public sources enter training, every stage file carries a manifest, and
every stage passes a 13-gram leakage gate.

Why: the first corpus builder walked local home folders (Desktop, Documents, Downloads, ...) and so pulled private
client documents into the MNTP / CGSA corpora and, through them, into the supervised and reranker data. Excluding
known names did not stop it, because nothing named the private folders. This module turns the rule around:

1. Allowlist (training/configs/sources_allowlist.json, schema defrost.sources/v1). Every source is listed with a
   pinned revision and a license: public OSS repos (https URL + 40-hex commit), public datasets (Hugging Face id +
   revision), public books (URL + commit + license). A file that does not sit under a listed source's clone, or a
   row whose source id is not listed, is rejected. There are no home-folder roots.
2. Denylist, a second guard. Generic private-path patterns live in the allowlist file. Client names, private paths
   and the private reference corpora live in a local, git-ignored file next to it (private_markers.local.json). A
   path or document that matches fails the build loudly. A strict build (the default) fails when that file is
   missing; DEFROST_PROVENANCE_STRICT=0 turns this off for public rebuilds.
3. Provenance manifest next to each stage file (<file>.provenance.json, schema defrost.provenance/v1): stage, file
   sha256 and size, and per source: id, kind, revision, license, docs, chars; plus the gate result.
4. Leakage gate on every stage: 13-gram overlap of each unit (document, line or row) with (a) the evaluation suites
   and (b) the private reference corpora. 13-grams that occur in two or more sources of the stage are boilerplate
   (license text, install snippets) and do not count. A unit with >= private_min private 13-grams fails the stage.
   A unit with >= eval_min eval 13-grams is dropped by the builders; `verify` fails on it.

    python -m defrost_graph.bilm.data_provenance check-sources      # allowlist schema; clones exist at the pinned commit
    python -m defrost_graph.bilm.data_provenance gate --stage mntp --file corpus_mntp.txt
    python -m defrost_graph.bilm.data_provenance verify corpus_mntp.txt.provenance.json

Gate reports hold counts and unit indices, never text, and are written to results/private/ (git-ignored)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SCHEMA = "defrost.provenance/v1"
SOURCES_SCHEMA = "defrost.sources/v1"
KINDS = {"oss", "hf", "book", "url"}
ROLES = {"train", "heldout", "dev"}
ALLOWLIST = "sources_allowlist.json"
MARKERS = "private_markers.local.json"
HEX40 = re.compile(r"[0-9a-f]{40}")
HOME_DIRS = re.compile(r"^~|(^|/)(Users|home)/|(^|/)(Desktop|Documents|Downloads|Dropbox[^/]*)(/|$)")


class SourceRejected(RuntimeError):
    """A file or row that does not come from an allowlisted source."""


class PrivateDataError(RuntimeError):
    """A path or document that matches a private marker, or a stage that overlaps the private corpora."""


class LeakageError(RuntimeError):
    """A stage file that overlaps an evaluation suite."""


# ── config ───────────────────────────────────────────────────────────────────

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists():
            return p
    return here.parents[-2]


def config_dir() -> Path:
    if os.environ.get("DEFROST_TRAINING_CONFIG"):
        return Path(os.environ["DEFROST_TRAINING_CONFIG"])
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        for cand in (base / "configs", base / "training/configs"):
            if (cand / ALLOWLIST).exists():
                return cand
    raise FileNotFoundError(f"{ALLOWLIST} not found (set DEFROST_TRAINING_CONFIG)")


def default_report_dir() -> Path:
    return Path(os.environ.get("DEFROST_PRIVATE_RESULTS") or repo_root() / "results/private") / "provenance"


class _Bases(dict):
    def __missing__(self, key):
        raise KeyError(f"no base directory {{{key}}} given to the allowlist (have: {sorted(self)})")


def expand(path: str, bases: dict) -> Path:
    """'{oss}/attrs', '{repo}/benchmarks/...', '~/x' or a repo-relative path -> absolute Path."""
    p = Path(os.path.expanduser(path.format_map(_Bases(bases))))
    return p if p.is_absolute() else Path(bases.get("repo", ".")) / p


def validate_spec(spec: dict) -> list[str]:
    out = []
    if spec.get("schema") != SOURCES_SCHEMA:
        out.append(f"schema must be {SOURCES_SCHEMA!r}")
    seen = set()
    for s in spec.get("sources", []):
        sid = s.get("id", "?")
        if sid in seen:
            out.append(f"duplicate id {sid}")
        seen.add(sid)
        for k in ("id", "kind", "role", "license", "revision"):
            if not s.get(k):
                out.append(f"{sid}: missing {k}")
        if s.get("kind") not in KINDS:
            out.append(f"{sid}: kind must be one of {sorted(KINDS)}")
        if s.get("role") not in ROLES:
            out.append(f"{sid}: role must be one of {sorted(ROLES)}")
        if s.get("kind") in ("oss", "book") and not (str(s.get("url", "")).startswith("https://")
                                                     and HEX40.fullmatch(str(s.get("revision", "")))):
            out.append(f"{sid}: oss/book sources need an https url and a pinned 40-hex commit")
        if s.get("kind") == "hf" and not (s.get("hf_id") and HEX40.fullmatch(str(s.get("revision", "")))):
            out.append(f"{sid}: hf sources need hf_id and a pinned 40-hex revision")
        if s.get("kind") == "url" and not (str(s.get("url", "")).startswith("http") and s.get("sha256")):
            out.append(f"{sid}: url sources need a url and the sha256 of the downloaded archive")
        if s.get("local_dir") and HOME_DIRS.search(s["local_dir"]):
            out.append(f"{sid}: local_dir must be a pinned clone under a base dir, not a home folder ({s['local_dir']})")
    return out


class Allowlist:
    """The allowlist plus the private markers. `bases` names the directories the spec's paths use, e.g.
    {"repo": <repo root>, "oss": <OSS clones>, "prose": <book clones>, "mega": <techdoc_mega>}."""

    def __init__(self, spec: dict, bases: dict | None = None, markers: dict | None = None, strict: bool = True):
        problems = validate_spec(spec)
        if problems:
            raise ValueError(f"{ALLOWLIST}: " + "; ".join(problems))
        self.spec = spec
        self.bases = {"repo": str(repo_root()), **{k: str(v) for k, v in (bases or {}).items()}}
        self.sources = {s["id"]: s for s in spec["sources"]}
        self.alias = {a: s["id"] for s in spec["sources"] for a in s.get("aliases", [])}
        self.gate_cfg = {"ngram": 13, "private_min": 3, "eval_min": 3, **spec.get("gate", {})}
        markers = markers or {}
        self.has_markers = bool(markers)
        self.deny = [re.compile(x, re.I) for x in spec.get("deny_path_patterns", []) + markers.get("path_patterns", [])]
        self.private_terms = [t.lower() for t in markers.get("content_terms", [])]
        self.private_corpora = markers.get("private_corpora", [])
        self.eval_kbs = markers.get("eval_kbs", [])          # private eval KBs (sqlite) for the eval 13-gram set
        if strict and not self.has_markers:
            raise PrivateDataError(f"strict build without {MARKERS}: create it next to {ALLOWLIST} (it is git-ignored) "
                                   "with path_patterns / content_terms / private_corpora, or set "
                                   "DEFROST_PROVENANCE_STRICT=0 for a public rebuild")

    @classmethod
    def load(cls, cfg: Path | None = None, bases: dict | None = None, strict: bool | None = None) -> "Allowlist":
        cfg = Path(cfg) if cfg else config_dir()
        spec = json.loads((cfg / ALLOWLIST).read_text())
        mf = cfg / MARKERS
        markers = json.loads(mf.read_text()) if mf.exists() else None
        if strict is None:
            strict = os.environ.get("DEFROST_PROVENANCE_STRICT", "1") != "0"
        return cls(spec, bases, markers, strict)

    def spec_sha(self) -> str:
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    def ids(self, kind: str | None = None, role: str | None = None) -> list[str]:
        return [i for i, s in self.sources.items()
                if (kind is None or s["kind"] == kind) and (role is None or s["role"] == role)]

    def resolve(self, name: str) -> str:
        """Source id for an id or an alias (e.g. the raw folder name 'peps' or the sup_data key 'msmarco')."""
        return name if name in self.sources else self.alias.get(name, name)

    def local_dir(self, sid: str) -> Path | None:
        s = self.sources[sid]
        return expand(s["local_dir"], self.bases).resolve() if s.get("local_dir") else None

    # guards --------------------------------------------------------------
    def check_path(self, path) -> None:
        """Second guard: a path that matches a denied pattern fails loudly."""
        s = str(path)
        for i, rx in enumerate(self.deny):
            if rx.search(s):
                raise PrivateDataError(f"denied path (pattern #{i}): {s}")

    def check_text(self, text: str, where: str = "") -> None:
        """Second guard on content: a private marker term in a document fails loudly (the term is not printed)."""
        low = text.lower()
        for i, t in enumerate(self.private_terms):
            if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low):
                raise PrivateDataError(f"private marker #{i} found in {where or 'a document'}")

    def source(self, name: str, roles=("train",)) -> dict:
        """The allowlisted source for an id / alias, or SourceRejected."""
        s = self.sources.get(self.resolve(name))
        if s is None:
            raise SourceRejected(f"not an allowlisted source: {name!r}")
        if s["role"] not in roles:
            raise SourceRejected(f"source {s['id']!r} has role {s['role']!r}, not one of {tuple(roles)}")
        return s

    def source_for_path(self, path, roles=("train",)) -> dict:
        """The allowlisted source whose pinned clone holds this file, or SourceRejected. Runs check_path first."""
        self.check_path(path)
        p = Path(path).expanduser().resolve()
        self.check_path(p)
        for sid in self.sources:
            try:
                d = self.local_dir(sid)
            except KeyError:
                continue
            if d is not None and (p == d or d in p.parents):
                return self.source(sid, roles)
        raise SourceRejected(f"not under any allowlisted source: {p}")


# ── 13-gram gate ─────────────────────────────────────────────────────────────

_WORD = re.compile(r"\w+")


def grams(text: str, n: int = 13) -> set[int]:
    """Stable (process-independent) hashes of the word 13-grams of a text."""
    w = _WORD.findall(text.lower())
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest(), "big")
            for i in range(len(w) - n + 1)}


TEXT_EXT = {".md", ".markdown", ".mdx", ".rst", ".txt", ".adoc"}


def texts_from(entry, bases: dict) -> list[str]:
    """Texts of one reference entry: a directory (text files, *.jsonl, *.sqlite), a .jsonl (question / answer /
    text), a .txt, or a memory sqlite (table sections). A dict entry {"path", "match"} keeps only files whose path
    relative to the directory matches the regex."""
    path, match = (entry["path"], entry.get("match")) if isinstance(entry, dict) else (entry, None)
    p = expand(path, bases)
    rx = re.compile(match, re.I) if match else None
    out = []
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if not f.is_file() or (rx and not rx.search(str(f.relative_to(p)))):
                continue
            if f.suffix.lower() in TEXT_EXT:
                out.append(f.read_text(errors="replace"))
            elif f.suffix in (".sqlite", ".jsonl"):
                out += texts_from(str(f), bases)
    elif p.suffix == ".sqlite" and p.exists():
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            out += [t for (t,) in con.execute("SELECT text FROM sections")]
        except sqlite3.OperationalError:
            pass
        con.close()
    elif p.suffix == ".jsonl" and p.exists():
        for line in open(p):
            r = json.loads(line)
            out += [str(r[k]) for k in ("question", "answer", "text") if r.get(k)]
    elif p.exists():
        out.append(p.read_text(errors="replace"))
    return out


def gram_set(entries: list, bases: dict, n: int = 13) -> tuple[set[int], list[dict]]:
    g, info = set(), []
    for e in entries:
        ts = texts_from(e, bases)
        before = len(g)
        for t in ts:
            g |= grams(t, n)
        info.append({"entry": e if isinstance(e, str) else e["path"], "texts": len(ts), "grams": len(g) - before})
    return g, info


_REF: dict = {}


def reference_grams(allow: Allowlist) -> tuple[set[int], set[int], dict]:
    """(eval 13-grams, private 13-grams, info), cached per process. A missing entry is reported (texts 0); when the
    markers file lists private corpora and none of them can be read, that is fatal."""
    key = (allow.spec_sha(), json.dumps(allow.private_corpora, sort_keys=True), json.dumps(allow.bases, sort_keys=True))
    if key not in _REF:
        n = allow.gate_cfg["ngram"]
        ev, ev_info = gram_set(allow.spec.get("eval_suites", []), allow.bases, n)
        pv, pv_info = gram_set(allow.private_corpora, allow.bases, n)
        if allow.private_corpora and not pv:
            raise PrivateDataError("the private reference corpora in the markers file are all missing or empty")
        _REF[key] = (ev, pv, {"eval": ev_info, "private": [{"entry": f"private#{i}", "texts": x["texts"],
                                                           "grams": x["grams"]} for i, x in enumerate(pv_info)]})
    return _REF[key]


def boilerplate(units: list[tuple[str, str]], n: int = 13) -> set[int]:
    """13-grams found in two or more different sources of the stage: license text, CI and install snippets. They
    are shared by unrelated projects, so they say nothing about where a document came from."""
    first: dict[int, str] = {}
    common: set[int] = set()
    for src, text in units:
        for h in grams(text, n):
            if first.setdefault(h, src) != src:
                common.add(h)
    return common


def gate(units: list[tuple[str, str]], eval_grams: set[int], private_grams: set[int], *, private_min: int = 3,
         eval_min: int = 3, ngram: int = 13, drop_eval: bool = True) -> dict:
    """units: (source id, text). A unit with >= private_min private 13-grams fails the stage. A unit with >= eval_min
    eval 13-grams is dropped (drop_eval) or fails. -> {"ok", "units", "kept", "private_hits", "eval_hits", ...};
    hits are (unit index, n 13-grams)."""
    common = boilerplate(units, ngram)
    priv, ev, kept = [], [], []
    for i, (_, text) in enumerate(units):
        g = grams(text, ngram) - common
        n_priv, n_eval = len(g & private_grams), len(g & eval_grams)
        if n_priv >= private_min:
            priv.append((i, n_priv))
        if n_eval >= eval_min:
            ev.append((i, n_eval))
            if drop_eval:
                continue
        kept.append(i)
    return {"ok": not priv and (drop_eval or not ev), "units": len(units), "kept": kept, "private_hits": priv,
            "eval_hits": ev, "boilerplate_grams": len(common), "private_min": private_min, "eval_min": eval_min,
            "ngram": ngram, "mode": "drop_eval" if drop_eval else "verify"}


def run_gate(stage: str, units: list[tuple[str, str]], allow: Allowlist, drop_eval: bool = True,
             report_dir: Path | None = None) -> dict:
    """Gate one stage against the allowlist's reference corpora, write the private report, raise on failure."""
    ev, pv, info = reference_grams(allow)
    c = allow.gate_cfg
    res = gate(units, ev, pv, private_min=c["private_min"], eval_min=c["eval_min"], ngram=c["ngram"],
               drop_eval=drop_eval)
    rep = Path(report_dir or default_report_dir())
    rep.mkdir(parents=True, exist_ok=True)

    def by_src(hits):
        return dict(Counter(units[i][0] for i, _ in hits))

    (rep / f"gate_{stage}.json").write_text(json.dumps({
        "stage": stage, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": res["ok"], "units": res["units"],
        "kept": len(res["kept"]), "private_hits": res["private_hits"], "private_hits_by_source": by_src(res["private_hits"]),
        "eval_hits": res["eval_hits"], "eval_hits_by_source": by_src(res["eval_hits"]), "mode": res["mode"],
        "boilerplate_grams": res["boilerplate_grams"], "reference": info, "thresholds": c}, indent=1))
    print(f"gate {stage}: {res['units']} units, private hits {len(res['private_hits'])}, eval hits "
          f"{len(res['eval_hits'])} ({'dropped' if drop_eval else 'fail'}) -> {'PASS' if res['ok'] else 'FAIL'} "
          f"(report: {rep / f'gate_{stage}.json'})", flush=True)
    if res["private_hits"]:
        raise PrivateDataError(f"stage {stage}: {len(res['private_hits'])} units overlap the private corpora "
                               f"(>= {c['private_min']} 13-grams); see {rep}")
    if not res["ok"]:
        raise LeakageError(f"stage {stage}: {len(res['eval_hits'])} units overlap an evaluation suite; see {rep}")
    return res


# ── manifests ────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_units(units: list[tuple[str, str]], keep: list[int] | None = None) -> dict:
    """-> {source id: {"n_docs", "n_chars"}} over the kept units."""
    out: dict = {}
    for i in (range(len(units)) if keep is None else keep):
        sid, text = units[i]
        c = out.setdefault(sid, {"n_docs": 0, "n_chars": 0})
        c["n_docs"] += 1
        c["n_chars"] += len(text)
    return out


def write_manifest(stage: str, file: Path, counts: dict, allow: Allowlist, gate_result: dict | None = None,
                   inputs: list[Path] | None = None, extra: dict | None = None) -> Path:
    """counts: source id -> {"n_docs", "n_chars"}; every id must be an allowlisted train source.
    inputs: upstream files (their sha256 is recorded, so a chain of stages can be checked end to end)."""
    file = Path(file)
    rows = []
    for sid, cnt in sorted(counts.items()):
        s = allow.source(sid)
        rows.append({"id": s["id"], "kind": s["kind"], "revision": s["revision"], "license": s["license"],
                     "license_verified": bool(s.get("license_verified", True)),
                     "origin": s.get("url") or f"https://huggingface.co/datasets/{s['hf_id']}",
                     "n_docs": int(cnt.get("n_docs", 0)), "n_chars": int(cnt.get("n_chars", 0))})
    g = None
    if gate_result is not None:
        g = {k: gate_result[k] for k in ("ok", "units", "boilerplate_grams", "private_min", "eval_min", "ngram", "mode")}
        g |= {"private_hits": len(gate_result["private_hits"]), "eval_hits": len(gate_result["eval_hits"])}
    m = {"schema": SCHEMA, "stage": stage, "file": file.name, "sha256": sha256(file), "bytes": file.stat().st_size,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "allowlist_sha256": allow.spec_sha(),
         "private_markers": allow.has_markers, "sources": rows, "gate": g,
         "inputs": [{"file": Path(p).name, "sha256": sha256(Path(p))} for p in inputs or []], **(extra or {})}
    out = file.with_name(file.name + ".provenance.json")
    out.write_text(json.dumps(m, indent=1))
    return out


def finalize_stage(stage: str, file: Path, units: list[tuple[str, str]], allow: Allowlist,
                   inputs: list[Path] | None = None, extra: dict | None = None, report_dir: Path | None = None) -> Path:
    """For builders that already wrote their stage file: check every row's source, gate in verify mode (any private
    or eval overlap fails), then write the manifest."""
    for sid in {u[0] for u in units}:
        allow.source(sid)
    res = run_gate(stage, units, allow, drop_eval=False, report_dir=report_dir)
    return write_manifest(stage, file, count_units(units), allow, res, inputs, extra)


def verify_manifest(path: Path, allow: Allowlist | None = None) -> list[str]:
    """Problems with a stage manifest: schema, file hash, gate passed, every source listed with license + revision
    (and, given the allowlist, still allowlisted at the same revision)."""
    path = Path(path)
    try:
        m = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"unreadable manifest: {e}"]
    out = []
    if m.get("schema") != SCHEMA:
        out.append(f"schema is not {SCHEMA}")
    for k in ("stage", "file", "sha256", "bytes", "created", "sources", "gate"):
        if k not in m:
            out.append(f"missing field {k}")
    f = path.with_name(m.get("file") or "")
    if not m.get("file") or not f.is_file():
        out.append(f"missing stage file {m.get('file')}")
    elif sha256(f) != m.get("sha256"):
        out.append("sha256 mismatch")
    g = m.get("gate") or {}
    if not g.get("ok") or g.get("private_hits", 1) != 0:
        out.append("leakage gate missing or failed")
    if not m.get("sources"):
        out.append("no sources")
    for s in m.get("sources", []):
        for k in ("id", "kind", "revision", "license", "n_docs", "n_chars"):
            if s.get(k) in (None, ""):
                out.append(f"{s.get('id')}: missing {k}")
        if allow is not None:
            try:
                a = allow.source(s.get("id", ""))
                if a["revision"] != s.get("revision"):
                    out.append(f"{s['id']}: revision differs from the allowlist")
            except SourceRejected as e:
                out.append(str(e))
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def check_sources(allow: Allowlist) -> list[str]:
    """Clones exist and sit at the pinned commit."""
    out = []
    for sid in allow.sources:
        try:
            d = allow.local_dir(sid)
        except KeyError:
            continue                                   # base dir not given (e.g. book clones for the corpus build)
        if d is None:
            continue
        if not d.exists():
            out.append(f"{sid}: clone missing at {d}")
            continue
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != allow.sources[sid]["revision"]:
            out.append(f"{sid}: clone at {head[:12] or '?'}, allowlist pins {allow.sources[sid]['revision'][:12]}")
    return out


def _units_from_file(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".jsonl":
        units = []
        for line in open(path):
            r = json.loads(line)
            parts = [r.get(k) for k in ("query", "positive", "negative", "text")] + list(r.get("negatives") or [])
            units.append((r.get("source_id", "?"), "\n".join(str(p) for p in parts if p)))
        return units
    return [("?", line) for line in open(path, errors="replace") if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", action="append", default=[], help="name=dir for the allowlist's {name} paths")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-sources")
    g = sub.add_parser("gate", help="verify-mode gate of one stage file (rows' source_id, or lines)")
    g.add_argument("--stage", required=True)
    g.add_argument("--file", required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifest", nargs="+")
    a = ap.parse_args(argv)
    bases = dict(x.split("=", 1) for x in a.base)
    allow = Allowlist.load(bases=bases, strict=a.cmd == "gate")
    if a.cmd == "check-sources":
        probs = check_sources(allow)
        print("\n".join(probs) or f"ok: {len(allow.sources)} sources")
        return 1 if probs else 0
    if a.cmd == "verify":
        bad = {m: verify_manifest(Path(m), allow) for m in a.manifest}
        for m, p in bad.items():
            print(f"{m}: {'ok' if not p else '; '.join(p)}")
        return 1 if any(bad.values()) else 0
    try:
        run_gate(a.stage, _units_from_file(Path(a.file)), allow, drop_eval=False)
    except (PrivateDataError, LeakageError) as e:
        print(f"FAIL: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
