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


def _port() -> str:
    return URL.rsplit(":", 1)[-1].split("/")[0]


def _token() -> str | None:
    from defrost_ai.service import auth
    st = auth.read_state(_port())
    return st.get("token") if st else None


def _call(method: str, path: str, body: dict | None = None, timeout: float = 600):
    headers = {"Content-Type": "application/json"}
    if (token := _token()):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _health() -> dict | None:
    try:
        return _call("GET", "/health", timeout=2)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def alive() -> bool:
    return bool((_health() or {}).get("ok"))


def _build() -> str | None:
    return (_health() or {}).get("build")


def _stop_by_port(port: str) -> None:
    """Terminate the local process listening on `port`, but only if its command line is a defrost service."""
    import signal
    try:
        pids = subprocess.run(["lsof", "-tiTCP:" + port, "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    except OSError:
        return
    for pid in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        if ("defrost" in cmd or "kev_memory" in cmd or "kev-memory" in cmd) and " serve" in cmd:
            os.kill(int(pid), signal.SIGTERM)
    t0 = time.time()
    while alive() and time.time() - t0 < 10:
        time.sleep(0.5)


def ensure_service(wait: float = 120) -> None:
    """Make sure a service answers on URL, starting one if needed.

    A running service is replaced only when it runs older code of THIS installation (same package path, different
    build: an upgrade) or is a pre-1.2 service without the install field. A service started by another installation
    (e.g. a repo .venv next to the uv tool) is left alone and used as it is, so two installs never keep restarting
    each other's service. DEFROST_SERVE_CMD set = the user manages the service: never restarted."""
    from defrost_ai import build_id
    from defrost_ai.service import auth
    health = _health()
    if health and health.get("ok"):
        same_code = health.get("build") == build_id()
        running = health.get("install")
        other_install = (running not in (None, auth.install_path())
                         and os.path.isdir(running))          # gone = this install, reinstalled (e.g. new Python)
        if same_code or other_install or os.environ.get("DEFROST_SERVE_CMD"):
            return
        try:                                            # our install, upgraded: restart the old service
            _call("POST", "/shutdown", {}, timeout=5)
        except (urllib.error.URLError, OSError):
            pass
        t0 = time.time()
        while alive() and time.time() - t0 < 10:
            time.sleep(0.5)
        if alive():                                     # pre-1.2 service (no token file / no install): stop by pid
            _stop_by_port(_port())
    port = _port()
    home = os.path.expanduser(os.environ.get("DEFROST_HOME", "~/.defrost-ai"))
    log = open(os.path.join(home, "service.log"), "a") if os.path.isdir(home) else subprocess.DEVNULL
    cmd = shlex.split(SERVE_CMD) if SERVE_CMD else [sys.executable, "-m", "defrost_ai.cli", "serve"]
    subprocess.Popen(cmd + ["--port", port], stdout=log, stderr=log, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        if alive() and _token():
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
