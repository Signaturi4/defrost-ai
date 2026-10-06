"""Query one built memory.

    from defrost_ai import Memory, Models
    models = Models()                                   # loads Defrost-Ret-B (+ Defrost-Rerank lazily); share across memories
    mem = Memory("~/.defrost-ai/acme", models)
    result = mem.search("how does the feed reach mobile", mode="accurate", k=5)   # or mode="fast"
    print(mem.context(result))                          # cited context pack for an LLM

Every hit is a section of a document, quoted verbatim with path and line range, plus the code nodes it names
(doc -> code links from the AST graph), so one query returns the text and the code it belongs to."""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from defrost_ai import conflicts, store
from defrost_ai.retrieval import keyword, policy

IDENTIFIER = re.compile(r"[a-z][A-Z]|_|[./]|\d|^[A-Z][A-Z0-9_]{3,}$")     # camelCase, snake_case, a.b, a/b, x2, CONST
PATH_OR_CALL = re.compile(r"(?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|jsx|mjs|go|rs|rb|java|kt|swift|sh|ya?ml|toml|sql|tf)"
                          r"|[A-Za-z_][\w.]*\(\)")


def _default_mode() -> str:
    from defrost_ai import settings
    return settings.get("search.mode")


class Models:
    """Loads the models once; the reranker only when a query first needs it (it is the slow part)."""

    def __init__(self, weights: Path | None = None, device: str | None = None):
        self.weights, self.device = weights, device
        self.encoder = None                                  # set by Memory to the encoder its vectors came from
        self._retriever = self._reranker = None

    @property
    def retriever(self):
        if self._retriever is None:
            from defrost_ai.models.encoder import encoder_name, make_retriever
            self._retriever = make_retriever(self.encoder or encoder_name(), self.weights, self.device)
        return self._retriever

    @property
    def reranker(self):
        if self._reranker is None:
            from defrost_ai.models.reranker import DefrostReranker
            self._reranker = DefrostReranker(self.weights, self.device)
        return self._reranker


class Memory:
    def __init__(self, out_dir: str | Path, models: Models | None = None, name: str | None = None):
        self.out = Path(out_dir).expanduser()
        self.models = models or Models()
        self.manifest = json.loads((self.out / store.MANIFEST).read_text())
        self.name = name or self.manifest.get("workspace", self.out.name)
        self.db = store.open_readonly(self.out / store.KNOWLEDGE_DB)
        v = np.load(self.out / store.SECTION_VECTORS)
        built_with = str(v["encoder"]) if "encoder" in v.files else "defrost-ret-b"
        if self.models.encoder not in (None, built_with):
            raise RuntimeError(f"memory {self.out} was built with encoder {built_with!r}, but the loaded models use "
                               f"{self.models.encoder!r}: rebuild it or switch retrieval.encoder")
        self.models.encoder = built_with                    # queries must use the encoder the sections used
        self.section_ids = list(v["ids"])
        self.section_vecs = v["vecs"].astype(np.float32)
        self.section_index = {s: i for i, s in enumerate(self.section_ids)}
        code = json.loads((self.out / store.CODE_GRAPH).read_text())
        self.code_nodes = {n["id"]: n for n in code["nodes"]}
        self.file_times = code.get("file_times", {})
        self.workspace_file = self.manifest.get("workspace_file")
        self.code_links = defaultdict(list)
        for sid, nid in self.db.execute("SELECT section_id, node_id FROM links WHERE confidence='EXTRACTED' ORDER BY rowid"):
            if nid in self.code_nodes:
                self.code_links[sid].append(nid)

    def section(self, sid: str) -> dict:
        row = self.db.execute(f"SELECT {', '.join(store.SECTION_FIELDS)} FROM sections WHERE id=?", (sid,)).fetchone()
        return dict(zip(store.SECTION_FIELDS, row))

    def candidates(self, query: str, qvec: np.ndarray | None = None):
        """-> (bm25 ids, dense ids, cosine of every candidate): the two first-stage lists, top 50 each."""
        bm25 = [s for s, _ in keyword.bm25_sections(self.db, query, policy.DEPTH)]
        qvec = self.models.retriever.embed_query(query) if qvec is None else qvec
        cos = self.section_vecs @ qvec
        top = np.argsort(-cos)[:policy.DEPTH]
        dense = [self.section_ids[i] for i in top]
        ix = self.section_index
        return bm25, dense, {s: float(cos[ix[s]]) for s in dict.fromkeys(bm25 + dense)}

    def rerank_scores(self, query: str, bm25: list[str], dense: list[str]) -> dict[str, float]:
        pool = policy.rerank_pool(bm25, dense)
        secs = [self.section(s) for s in pool]
        scores = self.models.reranker.score(query, [f"{s['heading_path']}\n{s['text']}" for s in secs])
        return {s: float(v) for s, v in zip(pool, scores)}

    def search(self, query: str, mode: str | None = None, k: int | str = 5, qvec: np.ndarray | None = None) -> dict:
        """-> {"query", "mode", "mode_used", "k", "hits": [...], "timing_ms"}. k="auto": adaptive k (1-5) from the
        calibrated dense confidence (policy.auto_k)."""
        t0 = time.time()
        asked = mode or _default_mode()
        mode = policy.normalize(asked)                  # accurate | fast -> internal policy
        bm25, dense, cos = self.candidates(query, qvec)
        t1 = time.time()
        scores = self.rerank_scores(query, bm25, dense) if policy.needs_reranker(mode, bm25, dense) else None
        t2 = time.time()
        ranked, used = policy.rank(mode, bm25, dense, scores)
        k = policy.auto_k(ranked, cos) if k == "auto" else int(k)
        from defrost_ai import trust
        level = trust.read(self.workspace_file)                 # read per search: a changed setting applies at once
        hits = []
        for i, sid in enumerate(ranked[:k]):
            s = self.section(sid)
            code = [{"label": self.code_nodes[n].get("label"), "file": self.code_nodes[n].get("source_file"),
                     "location": self.code_nodes[n].get("source_location"), "kind": self.code_nodes[n].get("kind")}
                    for n in self.code_links.get(sid, [])[:6]]
            verify, stale = self.grounding(sid, s["path"])
            missing = self.missing_names(sid, s["path"])
            hits.append({"rank": i + 1, "domain": self.name, "section_id": sid, "path": s["path"],
                         "lines": [s["line_start"], s["line_end"]], "heading": s["heading_path"], "text": s["text"],
                         "rerank_score": None if scores is None else scores.get(sid), "cosine": cos.get(sid),
                         "code": code, "verify": verify, "stale": stale, "missing": missing, "doc_trust": level})
        from defrost_ai.models import weights
        return {"query": query, "mode": asked, "mode_used": used, "k": k, "hits": hits,
                "timing_ms": {"first_stage": round(1000 * (t1 - t0)), "rerank": round(1000 * (t2 - t1))},
                "weights_warning": weights.WARNING}

    def grounding(self, sid: str, doc_path: str, n: int = 3) -> tuple[list[str], list[dict]]:
        """Files to check an answer against, and the ones where the doc is likely out of date.

        verify: the code and config files this section links to, at most `n`, project core and config files first
        (if the section names no file, the files its page links to).
        stale: a linked file counts only when its commits since the doc's last commit changed lines that contain a
        name this section mentions. "The file was committed after the doc" alone is true for almost every hit in an
        active repo (53% of hits on a product repo), so it is not shown."""
        core = tuple(self.manifest.get("core") or ())

        def rank(c):
            node = self.code_nodes[c]
            f = node.get("source_file") or ""
            return 0 if (core and f.startswith(core)) or node.get("kind") == "config_file" else 1

        def files_of(sids):
            nodes = [c for x in sids for c in self.code_links.get(x, [])]
            return list(dict.fromkeys(self.code_nodes[c]["source_file"] for c in sorted(nodes, key=rank)
                                      if self.code_nodes[c].get("source_file")))[:n]
        files = files_of([sid])
        if not files:                               # section names no file: fall back to what its page links to
            files = files_of([x for (x,) in self.db.execute("SELECT id FROM sections WHERE path=? ORDER BY ordinal",
                                                             (doc_path,))])
        row = self.db.execute("SELECT time FROM docs WHERE path=?", (doc_path,)).fetchone()
        doc_t = row[0] if row and row[0] else None
        stale = []
        newer = [f for f in files if doc_t and self.file_times.get(f, 0) > doc_t]
        if newer:
            names = self.section_names(sid)
            for f in newer:
                hit = sorted(n for n in names if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])",
                                                           self._changed_text(f, doc_t)))
                if hit:
                    stale.append({"file": f, "changed": time.strftime("%Y-%m-%d", time.localtime(self.file_times[f])),
                                  "doc": time.strftime("%Y-%m-%d", time.localtime(doc_t)), "names": hit[:4]})
        return files, stale

    def section_names(self, sid: str) -> set[str]:
        """Identifiers the section uses: its linked and unresolved code mentions and its backticked spans, split into
        parts (`a.b.c()` -> a.b.c, c; `deploy/backup.sh` -> backup.sh). Only identifier-shaped names count (camelCase,
        snake_case, CONSTANT, a dot, slash or digit): plain words such as `deploy` or `status` occur in almost any diff."""
        raw = [m for (m,) in self.db.execute("SELECT mention FROM links WHERE section_id=? UNION "
                                             "SELECT mention FROM unresolved WHERE section_id=?", (sid, sid))]
        row = self.db.execute("SELECT text FROM sections WHERE id=?", (sid,)).fetchone()
        if row:
            raw += re.findall(r"`([^`\n]{2,80})`", row[0])
        out = set()
        for m in raw:
            for part in re.split(r"\s+", m.strip().strip("`")):
                part = part.strip("()[]{},;:'\"").removesuffix("()").lstrip(".")
                for name in {part, part.rsplit(".", 1)[-1], part.rsplit("/", 1)[-1]}:
                    if len(name) >= 4 and re.fullmatch(r"[\w.$/-]+", name) and IDENTIFIER.search(name):
                        out.add(name)
        return out

    def _changed_text(self, rel: str, since: float) -> str:
        """Changed lines (+/-) of a file in commits after `since`, newest first, at most 30 commits / 200 KB.
        Cached per (file, doc time): every hit of one search reuses it."""
        key = (rel, since)
        cache = self.__dict__.setdefault("_changed", {})
        if key in cache:
            return cache[key]
        text = ""
        for root, sub in self._repo_paths(rel):
            try:
                r = subprocess.run(["git", "-C", str(root), "log", "-p", "-U0", "--no-color", "--format=", "-n", "30",
                                    f"--since={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(since + 1))}",
                                    "--", sub], capture_output=True, text=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            text = "\n".join(l[1:] for l in r.stdout[:200_000].splitlines()
                             if l[:1] in "+-" and not l.startswith(("+++", "---")))
            if r.returncode == 0:
                break
        cache[key] = text
        return text

    def code_evidence(self, sid: str, max_lines: int = 6) -> list[str]:
        """Code lines that back a section: for each identifier the section names (an EXTRACTED link), the lines of
        the linked file that contain it, as 'file:line: text'. Lets an agent check a doc against the code without
        opening files. File-name mentions are skipped (their lines say nothing about behaviour)."""
        out, seen = [], set()
        rows = self.db.execute("SELECT node_id, mention FROM links WHERE section_id=? AND confidence='EXTRACTED' "
                               "ORDER BY score DESC, rowid", (sid,)).fetchall()
        for nid, mention in rows:
            node = self.code_nodes.get(nid)
            name = (mention or "").strip("`").removesuffix("()")
            if not node or not node.get("source_file") or len(name) < 3 or "/" in name or \
                    name == Path(node["source_file"]).name:
                continue
            for root, sub in self._repo_paths(node["source_file"]):
                try:
                    lines = (root / sub).read_text(errors="replace").splitlines()
                except OSError:
                    continue
                for i, line in enumerate(lines, 1):
                    key = (sub, i)
                    if name in line and key not in seen:
                        seen.add(key)
                        out.append(f"{sub}:{i}: {line.strip()[:160]}")
                        break                                # first use per (name, file): the definition or the key line
                break
            if len(out) >= max_lines:
                break
        # Names that resolve to no code symbol (env variables, config keys, string constants, SQL names) are the
        # ones fact questions ask about; find their first use in code by exact word match.
        roots = list(dict.fromkeys(r for r, _ in self._repo_paths(self.name + "/x")))
        for (mention,) in self.db.execute("SELECT mention FROM unresolved WHERE section_id=? ORDER BY rowid", (sid,)):
            name = (mention or "").strip("`").removesuffix("()")
            if len(out) >= max_lines or len(name) < 4 or "/" in name or " " in name:
                continue
            for root in roots:
                try:
                    r = subprocess.run(["git", "-C", str(root), "grep", "-n", "-I", "-F", "-w", "-m", "1", "-e", name,
                                        "--", ".", ":!*.md", ":!*.lock", ":!*.json"],
                                       capture_output=True, text=True, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    continue
                for line in r.stdout.splitlines()[:1]:
                    f, ln, text = line.split(":", 2)
                    if (f, int(ln)) not in seen:
                        seen.add((f, int(ln)))
                        out.append(f"{f}:{ln}: {text.strip()[:160]}")
                break
        return out

    def _repo_paths(self, rel: str):
        """(repo root, path inside it) candidates for a display path '<component>/<path>'."""
        comp, _, sub = rel.partition("/")
        roots = {}
        try:
            spec = json.loads(Path(self.workspace_file).read_text()) if self.workspace_file else {}
            roots = {c["name"]: Path(c["path"]).expanduser() for c in spec.get("components", [])}
        except (OSError, ValueError, KeyError):
            pass
        if comp in roots:
            yield roots[comp], sub
        for root in self.manifest.get("sources", {}):
            if Path(root).name == comp:
                yield Path(root), sub

    def missing_names(self, sid: str, doc_path: str) -> list[str]:
        """Names the section uses that do not exist in the code: a call `name()`, or a file path whose directory
        exists in the repo (at its root or next to the doc) while the file does not. Strong signs that the doc
        describes code that was renamed or removed. Paths into other repos, env vars, URLs and prose words are left
        out: they are not expected to be in this code graph."""
        row = self.db.execute("SELECT abspath FROM docs WHERE path=?", (doc_path,)).fetchone()
        bases = [Path(r) for r in self.manifest.get("sources", {})] + ([Path(row[0]).parent] if row else [])
        out = []
        for (m,) in self.db.execute("SELECT mention FROM unresolved WHERE section_id=?", (sid,)):
            m = m.strip().strip("`")
            if not PATH_OR_CALL.fullmatch(m) or m.startswith(("http", "/")):
                continue
            if m.endswith("()") or any((b / m).parent.is_dir() and not (b / m).exists() for b in bases):
                out.append(m)
        calls = [m for m in out if m.endswith("()")]
        if calls:                                   # the AST misses some definitions (arrow consts, dynamic exports):
            found = self._in_code([c[:-2].rsplit(".", 1)[-1] for c in calls])     # a textual hit in code clears it
            out = [m for m in out if not (m.endswith("()") and m[:-2].rsplit(".", 1)[-1] in found)]
        bare = [m for m in out if "/" not in m and not m.endswith("()")]
        if bare:                                    # a bare `name.ext` is missing only when no tracked file has that
            tracked = self._tracked_names()         # name and the string does not occur in source code (`Prisma.sql`
            in_src = self._in_code(bare, source_only=True)                      # is an identifier, not a file)
            out = [m for m in out if m not in bare or (m not in tracked and m not in in_src)]
        return out[:5]

    def _tracked_names(self) -> set[str]:
        """Base names of every tracked file in the indexed repos (one `git ls-files` per repo, cached)."""
        if "_tracked" not in self.__dict__:
            names = set()
            for root in self.manifest.get("sources", {}):
                try:
                    r = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, timeout=10)
                    names |= {f.rsplit("/", 1)[-1] for f in r.stdout.splitlines()}
                except (OSError, subprocess.SubprocessError):
                    pass
            self.__dict__["_tracked"] = names
        return self.__dict__["_tracked"]

    SOURCE_GLOBS = [f"*.{e}" for e in ("py", "ts", "tsx", "js", "jsx", "mjs", "cjs", "go", "rs", "rb", "java", "kt",
                                       "swift", "php", "cs", "c", "cc", "cpp", "h")]

    def _in_code(self, names: list[str], source_only: bool = False) -> set[str]:
        """Which of `names` occur as whole words in non-doc files of the indexed repos (one `git grep` per repo);
        source_only: in program source files only (not config, yaml or env files, which often keep old names)."""
        found = set()
        for root in self.manifest.get("sources", {}):
            args = ["git", "-C", root, "grep", "-h", "-o", "-w", "-I", "-F"]
            for n in names:
                args += ["-e", n]
            args += ["--"] + (self.SOURCE_GLOBS if source_only else [".", ":!*.md", ":!*.mdx", ":!*.rst"])
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=5)
                found |= set(r.stdout.split())
            except (OSError, subprocess.SubprocessError):
                found |= set(names)                 # cannot check: do not claim a conflict
        return found

    def docs_for(self, path: str) -> list[dict]:
        """Doc sections that link to a code or config file (any node in it): the docs to review after changing it."""
        path = path.strip().removeprefix("./")
        nodes = [n for n, v in self.code_nodes.items()
                 if (v.get("source_file") or "").endswith("/" + path) or v.get("source_file") == path]
        if not nodes:
            return []
        q = f"SELECT DISTINCT l.section_id FROM links l WHERE l.node_id IN ({','.join('?' * len(nodes))})"
        out = []
        for (sid,) in self.db.execute(q, nodes):
            s = self.section(sid)
            out.append({"domain": self.name, "path": s["path"], "lines": [s["line_start"], s["line_end"]],
                        "heading": s["heading_path"]})
        return sorted(out, key=lambda h: (h["path"], h["lines"][0]))

    @staticmethod
    def context(result: dict, budget_tokens: int = 2000, words_per_hit: int = 300) -> str:
        """Cited context pack: one block per hit ('[n] path:Lx-y  heading' + text + linked code), ~4 chars/token."""
        from defrost_ai import trust
        parts, used = [], 0
        if result.get("weights_warning"):                   # older weights on purpose: say so on every answer
            parts.append(f"! {result['weights_warning']}\n")
        levels = {}
        for h in result["hits"]:
            levels.setdefault(h.get("doc_trust", trust.DEFAULT), []).append(h["domain"])
        for level, domains in levels.items():                   # what to do with these sections, set per project
            names = ", ".join(dict.fromkeys(domains))
            parts.append(f"[{names}] {trust.HEADER[level]}\n")
            used += len(parts[-1]) // 4
        ledgers: dict[str, list] = {}
        for h in result["hits"]:
            body = " ".join(h["text"].split()[:words_per_hit])
            code = "".join(f"\n  -> code {c['label']} ({c['file']}{':' + c['location'] if c['location'] else ''})"
                           for c in h["code"])
            code += "".join(f"\n  ! doc may be stale: {x['file']} changed {x['changed']}, after this doc ({x['doc']}), "
                            f"in lines with " + ", ".join(f"`{n}`" for n in x.get("names", []))
                            for x in h.get("stale", []))
            if h.get("missing"):
                code += "\n  ! doc/code conflict: names not found in the code: " + ", ".join(f"`{m}`" for m in h["missing"])
            for r in conflicts.for_section(ledgers.setdefault(h["domain"], conflicts.load(h["domain"])),
                                           h["path"], h["lines"])[:1]:
                code += (f"\n  resolved: {r['meaning']} (user, {r['at'][:10]})"
                         + (f": {r['note']}" if r["note"] else ""))
            if h.get("verify"):
                code += "\n  verify in: " + ", ".join(h["verify"])
            code += "".join(f"\n  code: {line}" for line in h.get("evidence", []))
            block = f"[{h['rank']}] {h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}  {h['heading']}\n{body}{code}\n"
            if used + len(block) // 4 > budget_tokens:
                break
            parts.append(block)
            used += len(block) // 4
        return "\n".join(parts)

    def status(self) -> dict:
        return {"name": self.name, "out": str(self.out), **{k: self.manifest.get(k) for k in
                ("built_at", "encoder", "counts", "sources", "changed")}}
