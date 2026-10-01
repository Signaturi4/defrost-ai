"""Stdlib-only client for the resident service (used by the MCP server, the graphify fork and agent tasks).
Starts the service in the background if it is not running."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

def _default_url() -> str:
    try:
        from defrost_ai import settings
        return f"http://127.0.0.1:{settings.get('service.port')}"
    except Exception:                                   # noqa: BLE001
        return "http://127.0.0.1:8765"


URL = os.environ.get("DEFROST_URL") or _default_url()
SERVE_CMD = os.environ.get("DEFROST_SERVE_CMD")        # e.g. "/path/.venv/bin/defrost serve"


def _call(method: str, path: str, body: dict | None = None, timeout: float = 600):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def alive() -> bool:
    try:
        return _call("GET", "/health", timeout=2).get("ok", False)
    except (urllib.error.URLError, OSError):
        return False


def _build() -> str | None:
    try:
        return _call("GET", "/health", timeout=2).get("build")
    except (urllib.error.URLError, OSError):
        return None


def _stop_by_port(port: str) -> None:
    """Terminate the local process listening on `port`, but only if it is a defrost service."""
    import signal
    try:
        pids = subprocess.run(["lsof", "-tiTCP:" + port, "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    except OSError:
        return
    for pid in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        if "defrost" in cmd or "defrost_ai" in cmd:
            os.kill(int(pid), signal.SIGTERM)
    t0 = time.time()
    while alive() and time.time() - t0 < 10:
        time.sleep(0.5)


def ensure_service(wait: float = 120) -> None:
    if alive():
        from defrost_ai import build_id
        if _build() == build_id() or os.environ.get("DEFROST_SERVE_CMD"):
            return                                      # same code, or a service the user manages explicitly
        try:                                            # upgraded package: restart the old service
            _call("POST", "/shutdown", {}, timeout=5)
        except (urllib.error.URLError, OSError):
            pass
        t0 = time.time()
        while alive() and time.time() - t0 < 10:
            time.sleep(0.5)
        if alive():                                     # pre-1.1 service without /shutdown: stop it by pid
            _stop_by_port(URL.rsplit(":", 1)[-1])
    port = URL.rsplit(":", 1)[-1]
    home = os.path.expanduser(os.environ.get("DEFROST_HOME", "~/.defrost-ai"))
    log = open(os.path.join(home, "service.log"), "a") if os.path.isdir(home) else subprocess.DEVNULL
    cmd = shlex.split(SERVE_CMD) if SERVE_CMD else [sys.executable, "-m", "defrost_ai.cli", "serve"]
    subprocess.Popen(cmd + ["--port", port], stdout=log, stderr=log, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        if alive():
            return
        time.sleep(1)
    raise RuntimeError(f"defrost service did not start on {URL}")


def search(query: str, domains=None, mode: str | None = None, k: int | str = 5, context: bool = True,
           merge: str | None = None):
    ensure_service()
    return _call("POST", "/search", {"query": query, "domains": domains, "mode": mode, "k": k, "context": context,
                                     "merge": merge})


def docs_for(paths: list[str], domains=None):
    ensure_service()
    return _call("POST", "/docs_for", {"paths": paths, "domains": domains})


def domains():
    ensure_service()
    return _call("GET", "/domains")


def update(domain: str | None = None, workspace: str | None = None, wait: bool = False, poll: float = 2.0):
    ensure_service()
    job = _call("POST", "/update", {"domain": domain, "workspace": workspace})
    if not wait:
        return job
    while True:
        state = _call("GET", f"/jobs/{job['job']}")
        if state["state"] in ("done", "failed"):
            return state
        time.sleep(poll)


def job(job_id: str):
    return _call("GET", f"/jobs/{job_id}")


def rollback(domain: str):
    ensure_service()
    return _call("POST", "/rollback", {"domain": domain})
