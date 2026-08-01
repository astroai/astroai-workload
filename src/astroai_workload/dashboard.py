"""Ray Dashboard URL resolution + local reverse proxy.

Two surfaces:

1. :func:`resolve_dashboard_url` — derive the Ray Dashboard URL for the current
   cluster from env, persisted state, or live ``canfar ps`` discovery. Mirrors
   ``orx-wire-compute.py``'s connect-URL derivation so every consumer agrees.

2. :class:`DashboardProxy` — a small local reverse proxy (stdlib ``http.server``
   + ``httpx``) that forwards the public manager connectURL ``/dashboard/`` path
   to a local port. This is how a **notebook / marimo** session (which cannot
   rely on CANFAR pod-to-pod networking) can embed the native Ray UI: the
   browser hits the proxy on the session's own port, the proxy fetches the
   public dashboard URL, and strips frame-busting headers so an iframe works.

Requires the ``astroai-workload[cluster]`` extra only (no Ray).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
# Frame-busting / CSP headers that would block embedding in notebook iframes.
_FRAME_HEADERS = frozenset({"x-frame-options", "content-security-policy", "frame-ancestors"})


def jobs_url_from_connect(connect_url: str) -> str:
    """Jobs API address derived from a manager session connect URL."""
    base = connect_url.rstrip("/") + "/"
    # Dashboard reverse-proxy exposes the Jobs API under /dashboard/
    return base + "dashboard"


def resolve_dashboard_url(address: str | None = None) -> str | None:
    """Resolve the Ray Dashboard / Jobs URL for the current cluster.

    Priority:
      1. Explicit ``address`` or ``ASTROAI_RAY_JOBS_ADDRESS`` /
         ``RAY_DASHBOARD_URL`` (set inside a ray-manager session).
      2. Persisted connect URL under ``~/.astroai/ray/clusters/*/connect-url``
         (written by ``astroai-workload cluster ensure``).
      3. Live ``canfar ps`` discovery of a Running/Pending ray-manager session.

    Returns ``None`` when nothing is resolvable.
    """
    explicit = address or os.environ.get("ASTROAI_RAY_JOBS_ADDRESS", "").strip()
    if not explicit:
        explicit = os.environ.get("RAY_DASHBOARD_URL", "").strip()
    if explicit:
        return explicit.strip()

    persisted = read_persisted_connect_url()
    if persisted:
        return jobs_url_from_connect(persisted)

    from astroai_workload.canfar_ops import CanfarOps

    try:
        ops = CanfarOps()
        if not ops.auth_status().authenticated:
            return None
        for row in ops.list_sessions():
            status = str(row.get("status") or "")
            if status not in {"Running", "Pending"}:
                continue
            image = str(row.get("image") or row.get("imageName") or "").lower()
            name = str(row.get("name") or "").lower()
            if "ray-manager" in image or name in {
                "raymgr",
                "orx-ray-stg",
                "ray-manager",
                "astroai-compute",
            }:
                connect = str(row.get("connectURL") or row.get("connectUrl") or "").strip()
                if connect:
                    return jobs_url_from_connect(connect)
    except Exception:  # noqa: BLE001 — discovery must never raise to callers
        return None
    return None


def read_persisted_connect_url() -> str | None:
    """Newest ``connect-url`` file under ~/.astroai/ray/clusters/*/."""
    clusters = Path.home() / ".astroai" / "ray" / "clusters"
    if not clusters.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for root in clusters.iterdir():
        if not root.is_dir():
            continue
        path = root / "connect-url"
        if not path.is_file():
            continue
        try:
            url = path.read_text(encoding="utf-8").strip()
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if url:
            candidates.append((mtime, url if url.endswith("/") else url + "/"))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def persist_connect_url(cluster_id: str, connect_url: str) -> Path:
    """Record a manager connect URL so later dashboard resolution finds it."""
    from astroai_workload.state_store import cluster_state_dir

    path = cluster_state_dir(cluster_id) / "connect-url"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(connect_url.rstrip("/") + "/", encoding="utf-8")
    return path


def _canfar_ps_json() -> list[dict[str, Any]]:
    if shutil.which("canfar") is None:
        return []
    try:
        proc = subprocess.run(
            ["canfar", "ps", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    raw = (proc.stdout or "").strip()
    if not raw:
        return []
    for marker in ("[", "{"):
        idx = raw.find(marker)
        if idx >= 0:
            raw = raw[idx:]
            break
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def dashboard_iframe_html(url: str, *, height: int = 900) -> str:
    """HTML snippet embedding the native Ray Dashboard in a notebook/marimo cell."""
    return (
        f'<iframe src="{url}" width="100%" height="{height}" '
        'style="border:1px solid #24332a;border-radius:8px;background:#fff" '
        'sandbox="allow-scripts allow-same-origin allow-forms allow-popups"></iframe>'
    )


class DashboardProxy:
    """Local reverse proxy → manager connectURL ``/dashboard/``.

    Runs a ``ThreadingHTTPServer`` on 127.0.0.1:*port*; every request is
    forwarded (method + body + filtered headers) to ``upstream + path`` and the
    response is streamed back with frame-busting headers stripped so the native
    Ray Dashboard can be embedded in a notebook/marimo iframe.

    WebSocket endpoints are not proxied (``http.server`` is HTTP-only); the
    Dashboard's core jobs/nodes/metrics views work over HTTP.
    """

    def __init__(self, upstream: str, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self.upstream = upstream.rstrip("/")
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _handler_factory(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class _Handler(BaseHTTPRequestHandler):
            def _forward(self) -> None:
                target = proxy.upstream + self.path
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in _HOP_BY_HOP and k.lower() not in _FRAME_HEADERS
                }
                headers["Host"] = httpx.URL(proxy.upstream).host
                body = None
                length = self.headers.get("Content-Length")
                if length:
                    body = self.rfile.read(int(length))
                try:
                    with httpx.Client(timeout=None, follow_redirects=False) as client:
                        resp = client.request(
                            self.command,
                            target,
                            headers=headers,
                            content=body,
                        )
                except httpx.HTTPError as exc:
                    self.send_error(502, f"upstream unreachable: {exc}")
                    return
                self.send_response(resp.status_code)
                for key, value in resp.headers.items():
                    if key.lower() not in _HOP_BY_HOP and key.lower() not in _FRAME_HEADERS:
                        self.send_header(key, value)
                self.end_headers()
                for chunk in resp.iter_bytes():
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break

            do_GET = _forward  # type: ignore[assignment]
            do_POST = _forward  # type: ignore[assignment]
            do_PUT = _forward  # type: ignore[assignment]
            do_PATCH = _forward  # type: ignore[assignment]
            do_DELETE = _forward  # type: ignore[assignment]
            do_HEAD = _forward  # type: ignore[assignment]
            do_OPTIONS = _forward  # type: ignore[assignment]

            def log_message(self, *args: Any) -> None:  # noqa: ANN001 — keep quiet
                return

        return _Handler

    def start(self) -> None:
        handler = self._handler_factory()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"
