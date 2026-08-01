"""HTTP client for a CANFAR ray-manager's /api/v1 endpoints.

Lets any AstroAI session (webterm, notebook, marimo, or an agent like
orx/hermes) drive cluster lifecycle over the manager's public connect URL
without running inside the manager pod. Uses the same auth path as the
``canfar`` CLI (client cert or bearer token) for HTTPS requests.

Requires the ``astroai-workload[cluster]`` extra only (no Ray).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx

PartialPolicy = Literal["fail_and_cleanup", "accept_partial", "continue_waiting"]


def _client_cert() -> tuple[str, str] | str | None:
    """Return the CANFAR client cert for httpx if one is available."""
    pem = os.environ.get("CANFAR_CLIENT_CERT", "").strip()
    if not pem:
        candidates = [
            Path.home() / ".ssl" / "cadcproxy.pem",
            Path(os.environ.get("HOME", "/tmp")) / ".ssl" / "cadcproxy.pem",
        ]
        pem = next((str(p) for p in candidates if p.is_file()), "")
    if pem:
        key = os.environ.get("CANFAR_CLIENT_KEY", "").strip()
        if key:
            return (pem, key)
        return pem
    return None


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("CANFAR_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


class ManagerClient:
    """Thin client over a ray-manager's public HTTP API."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        cert = _client_cert()
        headers = _auth_headers()
        headers.update(kwargs.pop("headers", {}))
        with httpx.Client(timeout=self.timeout, cert=cert, headers=headers) as client:
            return client.request(method, f"{self.base_url}{path}", **kwargs)

    def readyz(self) -> bool:
        try:
            resp = self._request("GET", "/readyz")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def healthz(self) -> bool:
        try:
            resp = self._request("GET", "/healthz")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def status(self) -> dict[str, Any]:
        resp = self._request("GET", "/api/v1/status")
        resp.raise_for_status()
        return resp.json()

    def auth_status(self) -> dict[str, Any]:
        resp = self._request("GET", "/api/v1/auth/status")
        resp.raise_for_status()
        return resp.json()

    def preflight(self, *, async_mode: bool = True) -> dict[str, Any]:
        query = "?async=1" if async_mode else ""
        resp = self._request("POST", f"/api/v1/preflight/run{query}")
        resp.raise_for_status()
        return resp.json()

    def create_cluster(
        self,
        *,
        name: str = "",
        worker_count: int = 2,
        cores: int = 1,
        ram_gb: int = 4,
        gpus: int = 0,
        min_joined: int | None = None,
        partial_policy: PartialPolicy = "accept_partial",
        require_preflight: bool = True,
        async_mode: bool = True,
    ) -> dict[str, Any]:
        body = {
            "name": name,
            "worker_count": worker_count,
            "cores": cores,
            "ram_gb": ram_gb,
            "gpus": gpus,
            "min_joined": min_joined,
            "partial_policy": partial_policy,
            "require_preflight": require_preflight,
        }
        query = "?async=1" if async_mode else ""
        resp = self._request("POST", f"/api/v1/cluster/create{query}", json=body)
        resp.raise_for_status()
        return resp.json()

    def stop_cluster(self) -> dict[str, Any]:
        resp = self._request("POST", "/api/v1/cluster/stop")
        resp.raise_for_status()
        return resp.json()

    def reconcile(self) -> dict[str, Any]:
        resp = self._request("POST", "/api/v1/cluster/reconcile")
        resp.raise_for_status()
        return resp.json()

    def clean_orphans(self) -> dict[str, Any]:
        resp = self._request("POST", "/api/v1/cluster/clean-orphans")
        resp.raise_for_status()
        return resp.json()

    def launch_worker(
        self, *, cores: int = 1, ram_gb: int = 4, gpus: int = 0, require_preflight: bool = True
    ) -> dict[str, Any]:
        body = {
            "cores": cores,
            "ram_gb": ram_gb,
            "gpus": gpus,
            "require_preflight": require_preflight,
        }
        resp = self._request("POST", "/api/v1/workers/launch", json=body)
        resp.raise_for_status()
        return resp.json()

    def destroy_worker(self, session_id: str) -> dict[str, Any]:
        resp = self._request("DELETE", f"/api/v1/workers/{session_id}")
        resp.raise_for_status()
        return resp.json()

    def destroy_all_workers(self) -> dict[str, Any]:
        resp = self._request("POST", "/api/v1/workers/destroy-all")
        resp.raise_for_status()
        return resp.json()

    def wait_ready(self, *, timeout_seconds: int = 600, poll_seconds: int = 5) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.readyz():
                return True
            time.sleep(poll_seconds)
        return False

    def wait_operation(
        self, *, timeout_seconds: int = 1800, poll_seconds: int = 10
    ) -> dict[str, Any]:
        """Poll /api/v1/status until no background operation is running."""
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.status()
            except httpx.HTTPError:
                time.sleep(poll_seconds)
                continue
            op = last.get("operation") or {}
            if not op.get("running"):
                return last
            time.sleep(poll_seconds)
        return last
