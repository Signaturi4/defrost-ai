"""Resident memory service: loads the models once and serves every domain over local HTTP (JSON).
Agents, the MCP server and the graphify fork are thin clients of this service.

    defrost serve --port 8765

    GET  /health                       {"ok": true, "build", "install"}   (no token needed)
    GET  /domains                      registered domains with build status
    POST /search   {"query", "domains"?, "mode"?="accurate"|"fast" (default: config), "k"?=5 | "auto", "merge"?, "context"?=false}
    POST /update   {"domain"} | {"workspace", "domain"?}   -> {"job"}: rebuild in the background (incremental)
    GET  /jobs/<id>                    {"state": queued|running|done|failed, "log", "manifest"?}
    POST /rollback {"domain"}          swap back to the previous build

Every other request needs `Authorization: Bearer <token>` from <DEFROST_HOME>/service-<port>.json; see
defrost_ai/service/auth.py for the Host / Origin / Content-Type rules.

Locks: model calls (query embedding, reranking, section encoding) share one lock, because there is one GPU / MPS
device. A build holds it only per encoding chunk, so searches run between chunks and during the CPU stages (code graph,
sections, links). One build per domain at a time (a second update of the same domain waits). A build writes a staging
directory and swaps it in; searches keep using the previous build until the swap, then reload it."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from defrost_ai import builder
from defrost_ai.service import auth
from defrost_ai.library import Library, read_registry, register
from defrost_ai.memory import Memory

from defrost_ai import build_id

BUILD = build_id()                                           # the code this process loaded, fixed at start

DEFAULT_PORT = 8765


def _k(v):
    return "auto" if v == "auto" else int(v)


class Service:
    def __init__(self):
        self.library = Library()
        self.gpu = threading.Lock()                          # model calls only (see module docstring)
        self.jobs: dict[str, dict] = {}
        self._build_locks: dict[str, threading.Lock] = {}
        self._build_locks_guard = threading.Lock()

    def build_lock(self, key: str) -> threading.Lock:
        with self._build_locks_guard:
            return self._build_locks.setdefault(key, threading.Lock())

    def search(self, body: dict) -> dict:
        with self.gpu:
            res = self.library.search(body["query"], body.get("domains"), body.get("mode"), _k(body.get("k", 5)),
                                      body.get("merge"))
        if body.get("context"):
            res["context"] = Memory.context(res, int(body.get("budget_tokens", 2000)))
        return res

    def update(self, body: dict) -> dict:
        domain = body.get("domain")
        workspace = body.get("workspace") or read_registry()["domains"].get(domain, {}).get("workspace")
        if not workspace:
            raise KeyError(f"unknown domain {domain!r} and no workspace given")
        if domain and body.get("workspace"):
            register(domain, workspace, body.get("description", ""))
        job = {"id": uuid.uuid4().hex[:12], "domain": domain, "workspace": workspace, "state": "queued", "log": [],
               "started": time.time()}
        self.jobs[job["id"]] = job

        def run():
            job["state"] = "waiting" if self.build_lock(domain or workspace).locked() else "running"
            try:
                with self.build_lock(domain or workspace):
                    job["state"] = "running"
                    job["manifest"] = {k: v for k, v in builder.build(workspace, self.library.models,
                                                                       log=job["log"].append,
                                                                       model_lock=self.gpu).items() if k != "doc_hashes"}
                job["state"] = "done"
            except Exception:                                    # noqa: BLE001 - reported to the caller
                job["state"], job["error"] = "failed", traceback.format_exc(limit=5)
            job["seconds"] = round(time.time() - job["started"], 1)
        threading.Thread(target=run, daemon=True).start()
        return {"job": job["id"]}

    def rollback(self, body: dict) -> dict:
        ws = read_registry()["domains"][body["domain"]]["workspace"]
        with self.build_lock(body["domain"]):
            return {"rolled_back": builder.rollback(ws)}


def make_handler(service: Service, port: int = DEFAULT_PORT, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _denied(self, method: str) -> bool:
            err = auth.check(method, self.path, self.headers, port, token)
            if err:
                self._send(err[0], {"error": err[1]})
            return bool(err)

        def do_GET(self):
            if self._denied("GET"):
                return
            try:
                if self.path == "/health":
                    return self._send(200, {"ok": True, "build": BUILD, "install": auth.install_path()})
                if self.path == "/domains":
                    return self._send(200, service.library.domains())
                if self.path.startswith("/jobs/"):
                    job = service.jobs.get(self.path.split("/")[-1])
                    return self._send(200 if job else 404, job or {"error": "no such job"})
                self._send(404, {"error": "not found"})
            except Exception as e:                               # noqa: BLE001
                self._send(500, {"error": str(e)})

        def do_POST(self):
            if self._denied("POST"):
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/shutdown":
                    self._send(200, {"ok": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                route = {"/search": service.search, "/update": service.update, "/rollback": service.rollback,
                         "/docs_for": lambda b: service.library.docs_for(b["paths"], b.get("domains"))}.get(self.path)
                if route is None:
                    return self._send(404, {"error": "not found"})
                self._send(200, route(body))
            except (KeyError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:                               # noqa: BLE001
                self._send(500, {"error": str(e)})

    return Handler


def warm_up(service: Service) -> None:
    """Load both models and run one tiny pass through each, so the first real query does not pay for model loading
    and GPU kernel compilation. Holds the GPU lock, so a search that arrives meanwhile simply waits for it."""
    try:
        with service.gpu:
            models = service.library.models
            models.retriever.embed_query("warm up")
            models.reranker.score("warm up", ["warm up section", "a second, longer warm up section of text"])
    except Exception as e:                                       # noqa: BLE001  (no weights yet: first search loads)
        print(f"warm-up skipped: {e}", flush=True)


def _gc_merged():
    """Drop merged-weights caches that the current weights no longer use (1.8 GB each); never fails the service."""
    try:
        from defrost_ai.models.merged_cache import gc_current
        gc_current(log=lambda m: print(m, flush=True))
    except Exception as e:                                          # noqa: BLE001
        print(f"defrost: merged-cache cleanup skipped: {e}", flush=True)


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
    service = Service()
    threading.Thread(target=_gc_merged, daemon=True).start()
    if os.environ.get("DEFROST_WARMUP", "1") != "0":
        threading.Thread(target=warm_up, args=(service,), daemon=True).start()
    token = auth.new_token()
    server = ThreadingHTTPServer((host, port), make_handler(service, port, token))     # binds (fails if port taken)
    auth.write_state(port, BUILD, token)
    print(f"defrost service on http://{host}:{port}  (domains: {sorted(read_registry()['domains'])}; "
          f"token in {auth.state_path(port)})", flush=True)
    try:
        server.serve_forever()
    finally:
        auth.clear_state(port, token)
