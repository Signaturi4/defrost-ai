"""On-disk layout of one built memory (one domain):

    <out>/manifest.json         what was built, from which commits, with which model; build statistics
    <out>/knowledge.sqlite      docs, sections (+ FTS5 BM25 index), doc->code links, unresolved mentions
    <out>/code_graph.json       graphify AST nodes/edges (empty when graphify is not installed)
    <out>/section_vectors.npz   Defrost-Ret-B vectors of every section (float16, row order = section rowid)
    <out>.previous/             the build before (defrost rollback swaps them)
    .<name>.cache/ (next to out) embedding cache (text sha256 -> vector) and graphify's per-file cache"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE docs (id TEXT PRIMARY KEY, component TEXT, path TEXT, abspath TEXT, sha256 TEXT, time INTEGER,
                   words INTEGER, title TEXT);
CREATE TABLE sections (id TEXT PRIMARY KEY, doc_id TEXT, component TEXT, path TEXT, heading_path TEXT, level INTEGER,
                       ordinal INTEGER, line_start INTEGER, line_end INTEGER, text TEXT, words INTEGER,
                       n_mentions INTEGER, n_resolved INTEGER, n_unresolved INTEGER);
CREATE VIRTUAL TABLE sections_fts USING fts5(heading_path, text, content='sections', content_rowid='rowid',
                                             tokenize='porter unicode61');
CREATE TABLE links (section_id TEXT, node_id TEXT, mention TEXT, confidence TEXT, score REAL);
CREATE TABLE unresolved (section_id TEXT, mention TEXT);
"""
SECTION_FIELDS = ["id", "path", "heading_path", "line_start", "line_end", "text", "component"]

KNOWLEDGE_DB = "knowledge.sqlite"
CODE_GRAPH = "code_graph.json"
SECTION_VECTORS = "section_vectors.npz"
MANIFEST = "manifest.json"


def cache_dir(out: Path) -> Path:
    return out.parent / f".{out.name}.cache"


def create(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
