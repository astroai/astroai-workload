"""MCP stdio server tests (no live Ray cluster)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from astroai_workload.mcp import SERVER_INFO, handle_message

ROOT = Path(__file__).resolve().parents[1]


def _rpc(method: str, params: dict | None = None, msg_id: int | None = 1) -> dict:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if msg_id is not None:
        msg["id"] = msg_id
    return msg


def test_initialize_handshake() -> None:
    resp = handle_message(_rpc("initialize", {"protocolVersion": "2024-11-05"}))
    assert resp is not None
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"] == SERVER_INFO
    assert "tools" in resp["result"]["capabilities"]


def test_initialized_notification_no_response() -> None:
    assert handle_message(_rpc("notifications/initialized", msg_id=None)) is None


def test_ping() -> None:
    resp = handle_message(_rpc("ping"))
    assert resp == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_invalid_request() -> None:
    resp = handle_message({"jsonrpc": "1.0", "method": "x"})
    assert resp["error"]["code"] == -32600


def test_method_not_found() -> None:
    resp = handle_message(_rpc("bogus"))
    assert resp["error"]["code"] == -32601


def test_tools_list_exposes_four_tools() -> None:
    resp = handle_message(_rpc("tools/list"))
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == ["cluster_ensure", "cluster_status", "cluster_scale", "dashboard_url"]
    for tool in resp["result"]["tools"]:
        assert "inputSchema" in tool
        assert "handler" not in tool  # handlers never leak over the wire


def test_tools_call_unknown_tool() -> None:
    resp = handle_message(_rpc("tools/call", {"name": "nope", "arguments": {}}))
    assert resp["error"]["code"] == -32602


def test_tools_call_cluster_status(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"cluster": {"phase": "Running"}, "joined_workers": 2}
    monkeypatch.setattr("astroai_workload.mcp.cluster_status_payload", lambda address=None: payload)
    resp = handle_message(_rpc("tools/call", {"name": "cluster_status", "arguments": {}}))
    assert "error" not in resp
    assert json.loads(resp["result"]["content"][0]["text"]) == payload


def test_unknown_notification_gets_no_response() -> None:
    """JSON-RPC: never respond to notifications (messages without an id)."""
    assert handle_message(_rpc("notifications/cancelled", msg_id=None)) is None


def test_tools_call_cluster_ensure_business_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**kwargs):
        raise RuntimeError("No manager found. Set ASTROAI_RAY_JOBS_ADDRESS or pass --address.")

    monkeypatch.setattr("astroai_workload.mcp.cluster_ensure_payload", _boom)
    resp = handle_message(_rpc("tools/call", {"name": "cluster_ensure", "arguments": {}}))
    assert resp["result"]["isError"] is True
    assert "No manager found" in resp["result"]["content"][0]["text"]


def test_tools_call_cluster_scale_invalid_args() -> None:
    resp = handle_message(
        _rpc("tools/call", {"name": "cluster_scale", "arguments": {"workers": "not-a-number"}})
    )
    assert resp["error"]["code"] == -32602


def test_tools_call_cluster_scale_requires_workers() -> None:
    """Omitting `workers` must NOT default to 0 (would destroy every worker)."""
    resp = handle_message(_rpc("tools/call", {"name": "cluster_scale", "arguments": {}}))
    assert resp["error"]["code"] == -32602
    assert "workers" in resp["error"]["message"]


def test_initialize_echoes_known_protocol_version() -> None:
    resp = handle_message(_rpc("initialize", {"protocolVersion": "2025-03-26"}))
    assert resp["result"]["protocolVersion"] == "2025-03-26"


def test_initialize_falls_back_for_unknown_protocol_version() -> None:
    resp = handle_message(_rpc("initialize", {"protocolVersion": "2099-01-01"}))
    assert resp["result"]["protocolVersion"] == "2024-11-05"


def test_tools_call_dashboard_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "astroai_workload.mcp.dashboard_url_payload", lambda address=None: "http://127.0.0.1:8265"
    )
    resp = handle_message(_rpc("tools/call", {"name": "dashboard_url", "arguments": {}}))
    assert json.loads(resp["result"]["content"][0]["text"]) == {
        "dashboard_url": "http://127.0.0.1:8265"
    }


def test_serve_stdio_subprocess_e2e() -> None:
    """Feed a real client conversation through `python -m ... mcp serve`."""
    lines = [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        ),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
        "not json",
    ]
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "astroai_workload.cli", "mcp", "serve"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    out = [json.loads(line) for line in proc.stdout.strip().splitlines()]
    by_id = {m["id"]: m for m in out}
    assert by_id[1]["result"]["serverInfo"]["name"] == "astroai-workload"
    names = [t["name"] for t in by_id[2]["result"]["tools"]]
    assert "cluster_ensure" in names and "dashboard_url" in names
    assert by_id[3]["result"] == {}
    # The malformed line produced a parse-error response with no id.
    assert any(m.get("error", {}).get("code") == -32700 for m in out)
