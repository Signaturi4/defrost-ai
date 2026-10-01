"""Local service access control and lock behaviour. No weights, no torch: the model work is faked."""
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DEFROST_HOME", str(tmp_path))
    monkeypatch.delenv("DEFROST_SERVICE_AUTH", raising=False)
    return tmp_path


def test_token_file_is_owner_only_and_written_after_bind(home):
    from defrost_ai.service import auth
    auth.write_state(8792, "b1", "tok")
    path = auth.state_path(8792)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    st = auth.read_state(8792)
    assert st["token"] == "tok" and st["port"] == 8792 and st["build"] == "b1" and st["install"] == auth.install_path()
    auth.clear_state(8792, "someone-else")                 # another service's token: keep the file
    assert path.exists()
    auth.clear_state(8792, "tok")
    assert not path.exists()


def test_request_rules(home):
    from defrost_ai.service.auth import check
    h = {"Host": "127.0.0.1:8792", "Content-Type": "application/json"}
    assert check("GET", "/health", {"Host": "127.0.0.1:8792"}, 8792, "t") is None
    assert check("POST", "/search", h, 8792, "t")[0] == 401
    assert check("POST", "/search", h | {"Authorization": "Bearer wrong"}, 8792, "t")[0] == 401
    assert check("POST", "/search", h | {"Authorization": "Bearer t"}, 8792, "t") is None
    assert check("POST", "/search", h | {"Authorization": "Bearer t", "Origin": "https://evil.example"}, 8792, "t")[0] == 403
    assert check("GET", "/health", {"Host": "127.0.0.1:8792", "Origin": "null"}, 8792, "t")[0] == 403
    assert check("POST", "/search", h | {"Authorization": "Bearer t", "Origin": "http://localhost:3000"}, 8792, "t") is None
    assert check("GET", "/health", {"Host": "evil.example:8792"}, 8792, "t")[0] == 403      # DNS rebinding
    assert check("POST", "/search", {"Host": "127.0.0.1:8792", "Content-Type": "text/plain",
                                     "Authorization": "Bearer t"}, 8792, "t")[0] == 415


class _FakeLibrary:
    def __init__(self):
        self.models = object()

    def search(self, query, domains, mode, k, merge):
        return {"query": query, "hits": [], "mode": mode or "accurate", "mode_used": "fast", "k": k}

    def domains(self):
        return {}


def _serve(home, monkeypatch):
    from defrost_ai.service import auth, http_server
    service = http_server.Service.__new__(http_server.Service)
    service.library, service.gpu, service.jobs = _FakeLibrary(), threading.Lock(), {}
    service._build_locks, service._build_locks_guard = {}, threading.Lock()
    token = auth.new_token()
    server = ThreadingHTTPServer(("127.0.0.1", 0), http_server.make_handler(service, 0, token))
    port = server.server_address[1]
    server.RequestHandlerClass = http_server.make_handler(service, port, token)
    auth.write_state(port, http_server.BUILD, token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return service, server, port, token


def _req(port, path, body=None, token=None, headers=None):
    h = {"Content-Type": "application/json"} | (headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                 data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_http_server_requires_token_and_local_origin(home, monkeypatch):
    service, server, port, token = _serve(home, monkeypatch)
    try:
        assert _req(port, "/health")[0] == 200
        assert _req(port, "/search", {"query": "q"})[0] == 401
        assert _req(port, "/domains")[0] == 401
        assert _req(port, "/search", {"query": "q"}, token, {"Origin": "https://evil.example"})[0] == 403
        code, res = _req(port, "/search", {"query": "q"}, token)
        assert code == 200 and res["query"] == "q"
        assert _req(port, "/shutdown", {})[0] == 401                     # a web page cannot stop the service
        monkeypatch.setattr("defrost_ai.service.client.URL", f"http://127.0.0.1:{port}")
        from defrost_ai.service import client
        assert client._call("POST", "/search", {"query": "via client"})["query"] == "via client"
    finally:
        server.shutdown()


def test_build_does_not_block_search_and_same_domain_builds_queue(home, monkeypatch):
    from defrost_ai.service import http_server
    service, server, port, token = _serve(home, monkeypatch)
    calls = []

    def slow_build(workspace, models, log=print, model_lock=None):
        calls.append(("start", workspace, time.time()))
        for _ in range(10):                                   # 1 s total, model lock held only per short chunk
            with model_lock:
                time.sleep(0.01)
            time.sleep(0.09)
        calls.append(("end", workspace, time.time()))
        return {"counts": {}}

    monkeypatch.setattr(http_server.builder, "build", slow_build)
    monkeypatch.setattr(http_server, "register", lambda *a, **k: None)
    try:
        ws = str(home / "ws.json")
        j1 = _req(port, "/update", {"domain": "d", "workspace": ws}, token)[1]["job"]
        j2 = _req(port, "/update", {"domain": "d", "workspace": ws}, token)[1]["job"]
        time.sleep(0.2)
        t0 = time.time()
        assert _req(port, "/search", {"query": "during build"}, token)[0] == 200
        assert time.time() - t0 < 0.5                         # not blocked by the running build
        deadline = time.time() + 10
        while time.time() < deadline and not all(service.jobs[j]["state"] == "done" for j in (j1, j2)):
            time.sleep(0.05)
        assert [service.jobs[j]["state"] for j in (j1, j2)] == ["done", "done"]
        starts = [t for kind, _, t in calls if kind == "start"]
        ends = [t for kind, _, t in calls if kind == "end"]
        assert starts[1] >= ends[0]                           # same domain: the second build waited for the first
    finally:
        server.shutdown()
