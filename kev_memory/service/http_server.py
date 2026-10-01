"""Resident memory service: loads the models once and serves every domain over local HTTP (JSON).
Agents, the MCP server and the graphify fork are thin clients of this service.

    kev-memory serve --port 8765

    GET  /health                       {"ok": true}
    GET  /domains                      registered domains with build status
    POST /search   {"query", "domains"?, "mode"?="fast", "k"?=5 | "auto", "merge"?="rerank", "context"?=false}
    POST /update   {"domain"} | {"workspace", "domain"?}   -> {"job"}: rebuild in the background (incremental)
    GET  /jobs/<id>                    {"state": queued|running|done|failed, "log", "manifest"?}
    POST /rollback {"domain"}          swap back to the previous build

Model calls are serialised with one lock (one GPU / MPS device); a build holds it while it encodes."""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kev_memory import builder
from kev_memory.library import Library, read_registry, register
from kev_memory.memory import Memory

from kev_memory import build_id

BUILD = build_id()                                           # the code this process loaded, fixed at start

DEFAULT_PORT = 8765


def _k(v):
    return "auto" if v == "auto" else int(v)


class Service:
    def __init__(self):
        self.library = Library()
        self.gpu = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def search(self, body: dict) -> dict:
        with self.gpu:
            res = self.library.search(body["query"], body.get("domains"), body.get("mode", "fast"), _k(body.get("k", 5)),
                                      body.get("merge", "rerank"))
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
            job["state"] = "running"
            try:
                with self.gpu:
                    job["manifest"] = {k: v for k, v in builder.build(workspace, self.library.models,
                                                                       log=job["log"].append).items() if k != "doc_hashes"}
                job["state"] = "done"
            except Exception:                                    # noqa: BLE001 - reported to the caller
                job["state"], job["error"] = "failed", traceback.format_exc(limit=5)
            job["seconds"] = round(time.time() - job["started"], 1)
        threading.Thread(target=run, daemon=True).start()
        return {"job": job["id"]}

    def rollback(self, body: dict) -> dict:
        ws = read_registry()["domains"][body["domain"]]["workspace"]
        return {"rolled_back": builder.rollback(ws)}


def make_handler(service: Service):
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

        def do_GET(self):
            try:
                if self.path == "/health":
                    return self._send(200, {"ok": True, "build": BUILD})
                if self.path == "/domains":
                    return self._send(200, service.library.domains())
                if self.path.startswith("/jobs/"):
                    job = service.jobs.get(self.path.split("/")[-1])
                    return self._send(200 if job else 404, job or {"error": "no such job"})
                self._send(404, {"error": "not found"})
            except Exception as e:                               # noqa: BLE001
                self._send(500, {"error": str(e)})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/shutdown":                    # local service only (binds 127.0.0.1)
                    self._send(200, {"ok": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                route = {"/search": service.search, "/update": service.update, "/rollback": service.rollback}.get(self.path)
                if route is None:
                    return self._send(404, {"error": "not found"})
                self._send(200, route(body))
            except (KeyError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:                               # noqa: BLE001
                self._send(500, {"error": str(e)})

    return Handler


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
    service = Service()
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"kev-memory service on http://{host}:{port}  (domains: {sorted(read_registry()['domains'])})", flush=True)
    server.serve_forever()
