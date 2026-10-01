"""Access control for the local service (stdlib only: imported by the MCP server and the graphify client too).

The service binds 127.0.0.1, but any web page open in a browser can still send requests to localhost. So:

- On start the service writes <DEFROST_HOME>/service-<port>.json (mode 0600): port, pid, build, install path and a
  random token. Only processes of the same user can read it.
- Every request except GET /health must send `Authorization: Bearer <token>` (else 401).
- Every request, /health included, must name the service in its Host header (127.0.0.1 / localhost; blocks DNS
  rebinding) and must not come from a foreign web origin (an Origin header other than localhost; else 403).
- POST bodies must be `Content-Type: application/json` (else 415): a browser cannot send that cross-origin without a
  CORS preflight, which the service never answers.

DEFROST_SERVICE_AUTH=0 turns the token check off (Host/Origin checks stay), e.g. for an old client during an upgrade."""
from __future__ import annotations

import hmac
import json
import os
import secrets
import time
from pathlib import Path

LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def home() -> Path:
    return Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser()


def state_path(port: int | str) -> Path:
    return home() / f"service-{port}.json"


def install_path() -> str:
    """Which installation a process runs (the package directory); two installs share one home but not this."""
    return str(Path(__file__).resolve().parents[1])


def new_token() -> str:
    return secrets.token_urlsafe(32)


def write_state(port: int, build: str, token: str) -> str:
    """Write the token file for a service that has bound `port` (owner-only permissions). Call it only after the bind
    succeeded: a second service that fails to bind must not replace the running service's token."""
    state = {"port": port, "pid": os.getpid(), "build": build, "install": install_path(), "token": token,
             "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = state_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(state, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return token


def read_state(port: int | str) -> dict | None:
    try:
        return json.loads(state_path(port).read_text())
    except (OSError, ValueError):
        return None


def clear_state(port: int, token: str) -> None:
    """Remove the token file on shutdown, unless another service has already replaced it."""
    st = read_state(port)
    if st and st.get("token") == token:
        state_path(port).unlink(missing_ok=True)


def required() -> bool:
    return os.environ.get("DEFROST_SERVICE_AUTH", "1") != "0"


def _is_local_host(host: str, port: int) -> bool:
    name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
    return name in LOCAL_HOSTS


def _is_local_origin(origin: str) -> bool:
    if origin in ("", "null"):
        return origin == ""
    for scheme in ("http://", "https://"):
        if origin.startswith(scheme):
            rest = origin[len(scheme):]
            name = rest.split("]")[0] + "]" if rest.startswith("[") else rest.split(":")[0].split("/")[0]
            return name in LOCAL_HOSTS
    return False


def check(method: str, path: str, headers, port: int, token: str | None) -> tuple[int, str] | None:
    """None when the request may proceed, else (HTTP status, reason)."""
    if not _is_local_host(headers.get("Host", ""), port):
        return 403, "Host must be 127.0.0.1 or localhost"
    origin = headers.get("Origin")
    if origin is not None and not _is_local_origin(origin):
        return 403, "cross-origin requests are not allowed"
    if method == "POST" and headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
        return 415, "POST bodies must be application/json"
    if path == "/health" and method == "GET":
        return None
    if required() and token:
        sent = headers.get("Authorization", "")
        if not (sent.startswith("Bearer ") and hmac.compare_digest(sent[7:].strip(), token)):
            return 401, f"missing or wrong token (see {state_path(port)})"
    return None
