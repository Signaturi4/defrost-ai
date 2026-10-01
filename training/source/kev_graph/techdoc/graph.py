"""Deterministic code graph + folder hierarchy (P0 of docs/jev_for_graph/v2/tech_doc_graph_training_plan.md):
dataclasses, file discovery, build()/update()/changed_since(), and JSON serialization. Built with stdlib `ast`
-- not tree-sitter, which is not a dependency of this repo and only Python repos are in scope, so `ast` gives
everything the plan's P0 table asks for (files, classes, functions, imports, calls, inheritance, CLI flags, env
vars) with no new dependency.

Two-pass design, driven by what P0's incremental-update gate needs ("parsing is what update() saves, not
resolution"): pass 1 (py_extract.py) parses one file at a time into nodes (`contains`/`defines_flag`/`uses_env`
edges, which are intra-file) plus a small set of "raw facts" (call sites, base-class references, resolved import
specs, per-function locally-bound names) that get stored on the graph (`Graph.file_facts`), since resolving a
call or import may need a *different* file's symbol table that has not been parsed yet. Pass 2 (resolve.py)
redoes `imports`/`calls`/`inherits` globally from the current node index + stored per-file facts -- cheap,
because it never re-parses source. `update()` reparses only the given files and then reruns pass 2 over
everything, which is what makes it fast on a normal-sized diff.

`_build_files()` and `_resolve()` import py_extract.py / resolve.py LAZILY (inside the function body, not at
module load time): those two modules each import a handful of names FROM this module (`Node`, `_module_dotted`,
`_import_target_parts`), so a top-level import in both directions would be a circular import. The lazy import
breaks the cycle without changing where the dataclasses/parsing/resolution logically live.

Doc->code links (`kev_graph.techdoc.doc_links.exact_links`) are the free half of the plan's distant supervision
(section 4.1); see that module."""
import dataclasses
import fnmatch
import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 5

# Generic, repo-agnostic default excludes only -- anything repo-specific (this repo's `runs/`, `kev_graph/data/`,
# `kev_graph/agent_envoirment_ml/`, the untracked `kev_graph copy/`) is a --exclude the caller passes, not a
# built-in default (build() must work the same way on any repo).
DEFAULT_EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                         ".history"}


def _dir_excluded(name):
    return name in DEFAULT_EXCLUDE_DIRS or name.startswith(".venv")


def _exclude_match(relpath, patterns):
    """--exclude patterns match by relpath PREFIX (a whole directory subtree, e.g. "runs" excludes everything
    under runs/) or by fnmatch glob against the full relpath -- never by bare directory name at an arbitrary
    depth (that's what DEFAULT_EXCLUDE_DIRS is for, and conflating the two would make a `--exclude data` also
    swallow an unrelated `pkg/data/model.py`)."""
    for pat in patterns:
        pat = pat.rstrip("/")
        if not pat:
            continue
        if relpath == pat or relpath.startswith(pat + "/"):
            return True
        if fnmatch.fnmatch(relpath, pat):
            return True
    return False


def _is_included(relpath, exclude=()):
    """Single inclusion filter used by BOTH discovery (_iter_included_files) and update() -- so a caller handing
    update() an arbitrary changed-path list (P5: `git diff --name-only`, any file type, absolute or relative,
    including paths that don't exist on disk anymore) can never add a node build() itself would never have
    created."""
    if not (relpath.endswith(".py") or relpath.endswith(".md")):
        return False
    for part in Path(relpath).parts[:-1]:
        if _dir_excluded(part):
            return False
    return not _exclude_match(relpath, exclude)


def _normalize_relpath(raw, repo_root):
    """Normalise any path a caller might hand update() -- relative, absolute, with `..` segments, a different
    cwd -- to a posix relpath under `repo_root`. None if it doesn't resolve to something inside repo_root."""
    repo_root = Path(repo_root).resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    p = p.resolve()
    try:
        return p.relative_to(repo_root).as_posix()
    except ValueError:
        return None


@dataclass
class Node:
    id: str
    kind: str                    # dir, file, doc, class, function, cli_flag, env_var
    name: str
    path: str = None             # relpath (posix), or None for dir:. 's ".", or env vars
    lineno: int = None
    end_lineno: int = None
    sha: str = None              # commit this node was (re)built from -- R8, per-node freshness
    error: str = None            # set on file/doc nodes that failed to read/parse; not fatal


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)        # id -> Node
    edges: set = field(default_factory=set)           # {(src, dst, kind)}
    file_hashes: dict = field(default_factory=dict)   # relpath -> sha256(bytes), for changed_since()/update()
    file_facts: dict = field(default_factory=dict)    # relpath -> raw facts needed to re-resolve edges (see above)
    built_from_sha: str = None


# --------------------------------------------------------------------------------------------------------------
# git / hashing / file discovery
# --------------------------------------------------------------------------------------------------------------

def _git_head_sha(repo_root):
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_ls_files(repo_root):
    """Tracked + untracked-not-ignored files, relative to repo_root -- respects .gitignore for free, and is
    faster than an os.walk over a repo whose ignored directories (build output, venvs, model checkpoints) were
    never meant to be scanned. None (not a git repo, or git unavailable) falls back to _iter_included_files's
    os.walk branch.

    `-z` (NUL-separated, unquoted) instead of git's default newline-separated output: without it, git C-quotes
    any path with a non-ASCII or otherwise "unusual" byte (e.g. `pkg/caf\\303\\251.py`), which would silently
    turn into a bogus relpath -- and therefore a bogus node id -- for any real file with such a name."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "ls-files", "-z", "-co", "--exclude-standard", "--", "."],
                              capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return [p for p in out.stdout.split("\0") if p]


def _iter_included_files(repo_root, exclude=()):
    """.py and .md files under repo_root that pass _is_included AND still exist on disk.

    The disk check matters specifically for the git-listed path: `git ls-files -c` lists INDEX entries, which
    still show a file that was `rm`'d (not `git rm`'d) from the working tree on a dirty checkout -- without
    filtering those out, a fresh build() on such a tree would create error="unreadable" nodes (and dangling
    contains/imports edges to them) that build()+update() from the SAME state would never produce, since
    update() only re-adds a changed path when it `.is_file()`."""
    repo_root = Path(repo_root)
    listed = _git_ls_files(repo_root)
    if listed is not None:
        results = [p for p in listed if _is_included(p, exclude) and (repo_root / p).is_file()]
    else:
        results = []
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [d for d in dirnames if not _dir_excluded(d)]
            for fn in filenames:
                relpath = (Path(dirpath) / fn).relative_to(repo_root).as_posix()
                if _is_included(relpath, exclude):
                    results.append(relpath)
    return sorted(set(results))


def _module_dotted(relpath):
    """Dotted import path for a repo-relative .py path -- `pkg/sub/__init__.py` and `pkg/sub.py` both -> `pkg.sub`
    (a package and a same-named module can't coexist on disk, so this is unambiguous)."""
    parts = list(Path(relpath).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _import_target_parts(package_parts, level, module):
    """Resolve an (level, module) ImportFrom spec to dotted-path parts, given the importing file's own package
    parts. level=0 is absolute. level=1 is "from .x import y" (same package); each extra dot goes up one more
    package, matching Python's own relative-import semantics. Returns None (unresolvable, don't clamp) when
    `level` climbs past the file's own package depth -- e.g. `from ... import x` in a top-level module."""
    if level == 0:
        base = []
    else:
        cut = len(package_parts) - (level - 1)
        if cut < 0:
            return None
        base = package_parts[:cut]
    if module:
        return base + module.split(".")
    return base


# --------------------------------------------------------------------------------------------------------------
# directory chain / file removal / pruning (shared by build() and update())
# --------------------------------------------------------------------------------------------------------------

def _ensure_dir_chain(graph, relpath, sha):
    root_id = "dir:."
    if root_id not in graph.nodes:
        graph.nodes[root_id] = Node(id=root_id, kind="dir", name=".", path=".", sha=sha)
    parts = Path(relpath).parts[:-1]
    parent_id = root_id
    cur = []
    for part in parts:
        cur.append(part)
        rel = "/".join(cur)
        node_id = f"dir:{rel}"
        if node_id not in graph.nodes:
            graph.nodes[node_id] = Node(id=node_id, kind="dir", name=part, path=rel, sha=sha)
        graph.edges.add((parent_id, node_id, "contains"))
        parent_id = node_id
    return parent_id


def _build_files(graph, repo_root, relpaths, sha):
    from kev_graph.techdoc import py_extract  # lazy: breaks the graph.py <-> py_extract.py import cycle

    repo_root = Path(repo_root)
    for relpath in relpaths:
        parent_id = _ensure_dir_chain(graph, relpath, sha)
        full = repo_root / relpath
        try:
            data = full.read_bytes()
        except Exception:
            data = None

        if data is None:
            graph.file_hashes[relpath] = ""
            kind = "file" if relpath.endswith(".py") else "doc"
            graph.nodes[relpath] = Node(id=relpath, kind=kind, name=Path(relpath).name, path=relpath, sha=sha,
                                         error="unreadable")
            if relpath.endswith(".py"):
                graph.file_facts[relpath] = py_extract.empty_facts()
        else:
            graph.file_hashes[relpath] = hashlib.sha256(data).hexdigest()
            if relpath.endswith(".py"):
                try:
                    source = data.decode("utf-8")
                except UnicodeDecodeError as e:
                    graph.nodes[relpath] = Node(id=relpath, kind="file", name=Path(relpath).name, path=relpath,
                                                 sha=sha, error=f"UnicodeDecodeError: {e}")
                    graph.file_facts[relpath] = py_extract.empty_facts()
                else:
                    py_extract.parse_python_source(graph, relpath, source, sha)
            else:
                # .md is not parsed for content -- only decoded elsewhere (exact_links) when actually needed.
                graph.nodes[relpath] = Node(id=relpath, kind="doc", name=Path(relpath).name, path=relpath, sha=sha)

        graph.edges.add((parent_id, relpath, "contains"))


def _remove_file(graph, relpath):
    ids = {relpath}
    for node in graph.nodes.values():
        if node.path == relpath and node.kind in ("class", "function", "cli_flag"):
            ids.add(node.id)
    for nid in ids:
        graph.nodes.pop(nid, None)
    graph.edges = {e for e in graph.edges if e[0] not in ids and e[1] not in ids}
    graph.file_hashes.pop(relpath, None)
    graph.file_facts.pop(relpath, None)


def _prune_empty(graph):
    """Drop directory nodes left with no children and env_var nodes left with no `uses_env` edge, after a
    file was removed or changed. Iterates to a fixpoint since pruning a leaf dir can empty its parent."""
    changed = True
    while changed:
        changed = False
        has_children = {src for (src, _dst, kind) in graph.edges if kind == "contains"}
        for node_id in list(graph.nodes):
            node = graph.nodes[node_id]
            if node.kind == "dir" and node_id != "dir:." and node_id not in has_children:
                del graph.nodes[node_id]
                graph.edges = {e for e in graph.edges if e[0] != node_id and e[1] != node_id}
                changed = True
        used_env = {dst for (_src, dst, kind) in graph.edges if kind == "uses_env"}
        for node_id in list(graph.nodes):
            node = graph.nodes[node_id]
            if node.kind == "env_var" and node_id not in used_env:
                del graph.nodes[node_id]
                changed = True


def _resolve(graph):
    from kev_graph.techdoc import resolve  # lazy: breaks the graph.py <-> resolve.py import cycle
    resolve.resolve_edges(graph)


# --------------------------------------------------------------------------------------------------------------
# public API: build / update / changed_since
# --------------------------------------------------------------------------------------------------------------

def build(repo_root, exclude=()):
    repo_root = Path(repo_root)
    graph = Graph(built_from_sha=_git_head_sha(repo_root))
    relpaths = _iter_included_files(repo_root, exclude)
    _build_files(graph, repo_root, relpaths, graph.built_from_sha)
    _resolve(graph)
    _prune_empty(graph)
    return graph


def update(graph, repo_root, changed_paths, exclude=()):
    """Reparse only `changed_paths` (added/modified/deleted/renamed -- deleted/renamed-away paths simply no
    longer exist on disk), then redo global edge resolution over the whole graph. Any file NOT in
    `changed_paths` whose own content is unchanged is never reparsed, and keeps its old node `sha`.

    `changed_paths` may be anything a caller like `git diff --name-only` produces: relative or absolute, any
    file type, paths that no longer exist. Each is normalised to a repo-relative posix path and always removed
    from the graph if present; it is only re-added when it passes the SAME `_is_included` filter build() uses
    and still exists on disk -- so a non-.py/.md file or an excluded path can shrink the graph but never grow it
    with something build() itself would never have created."""
    repo_root = Path(repo_root).resolve()
    sha = _git_head_sha(repo_root)
    to_add = []
    for raw in changed_paths:
        relpath = _normalize_relpath(raw, repo_root)
        if relpath is None:
            continue
        _remove_file(graph, relpath)
        if _is_included(relpath, exclude) and (repo_root / relpath).is_file():
            to_add.append(relpath)
    _build_files(graph, repo_root, to_add, sha)
    graph.built_from_sha = sha
    _resolve(graph)
    _prune_empty(graph)
    return graph


def changed_since(graph, repo_root, exclude=()):
    """Files whose content hash differs from what's stored in `graph.file_hashes`, plus new and deleted files --
    for a post-merge hook to pass straight to `update()`."""
    repo_root = Path(repo_root)
    current = set(_iter_included_files(repo_root, exclude))
    known = set(graph.file_hashes)
    changed = sorted(current - known) + sorted(known - current)
    for relpath in sorted(current & known):
        full = repo_root / relpath
        try:
            data = full.read_bytes()
        except Exception:
            changed.append(relpath)
            continue
        if hashlib.sha256(data).hexdigest() != graph.file_hashes.get(relpath):
            changed.append(relpath)
    return changed


# --------------------------------------------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------------------------------------------

def to_json(graph):
    return {
        "schema_version": SCHEMA_VERSION,
        "built_from_sha": graph.built_from_sha,
        "nodes": [dataclasses.asdict(n) for n in sorted(graph.nodes.values(), key=lambda n: n.id)],
        "edges": sorted(list(e) for e in graph.edges),
        "file_hashes": dict(sorted(graph.file_hashes.items())),
        "file_facts": dict(sorted(graph.file_facts.items())),
    }


def from_json(data):
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"code_graph: unsupported schema_version {version!r} (expected {SCHEMA_VERSION!r}); "
                          f"rebuild the graph")
    g = Graph()
    g.built_from_sha = data.get("built_from_sha")
    for nd in data.get("nodes", []):
        g.nodes[nd["id"]] = Node(**nd)
    g.edges = {tuple(e) for e in data.get("edges", [])}
    g.file_hashes = dict(data.get("file_hashes", {}))
    g.file_facts = dict(data.get("file_facts", {}))
    return g
