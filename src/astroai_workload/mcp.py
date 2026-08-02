"""Minimal MCP (Model Context Protocol) server over stdio for Ray cluster ops.

Exposes the cluster lifecycle (ensure/status/scale) and dashboard URL as MCP
tools so MCP-capable agents (hermes, openclaw, Cursor, ...) can drive a CANFAR
Ray cluster through ``astroai-workload mcp serve`` — no Ray install required
(``[cluster]`` extra only).

Transport: MCP stdio — newline-delimited JSON-RPC 2.0 messages on stdin/stdout.
Deliberately zero-dependency (stdlib ``json`` only) to keep lean images lean;
the protocol surface is just ``initialize`` / ``notifications/initialized`` /
``ping`` / ``tools/list`` / ``tools/call``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from astroai_workload import __version__
from astroai_workload.cli import (
    cluster_ensure_payload,
    cluster_scale_payload,
    cluster_status_payload,
    dashboard_url_payload,
)

PROTOCOL_VERSION = "2024-11-05"
# Client-negotiated versions with an identical tools surface: echo the client's
# requested version when known so newer MCP clients (2025-03-26/2025-06-18)
# don't reject the server, else fall back to the base version.
_SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
SERVER_INFO = {"name": "astroai-workload", "version": __version__}

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _json_result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _json_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def _tool_cluster_ensure(args: dict[str, Any]) -> dict[str, Any]:
    return cluster_ensure_payload(
        address=args.get("address"),
        workers=int(args.get("workers", 0)),
        cores=int(args.get("cores", 1)),
        ram=int(args.get("ram", 4)),
        gpus=int(args.get("gpus", 0)),
        timeout=int(args.get("timeout", 1800)),
    )


def _tool_cluster_status(args: dict[str, Any]) -> dict[str, Any]:
    return cluster_status_payload(args.get("address"))


def _tool_cluster_scale(args: dict[str, Any]) -> dict[str, Any]:
    # `workers` is required — omitting it must NOT default to 0 (that would
    # destroy every worker session). The CLI makes it a positional argument.
    if "workers" not in args:
        raise ValueError("workers is required")
    return cluster_scale_payload(
        workers=int(args["workers"]),
        address=args.get("address"),
        cores=int(args.get("cores", 1)),
        ram=int(args.get("ram", 4)),
        gpus=int(args.get("gpus", 0)),
        timeout=int(args.get("timeout", 1800)),
    )


def _tool_dashboard_url(args: dict[str, Any]) -> dict[str, Any]:
    url = dashboard_url_payload(args.get("address"))
    return {"dashboard_url": url}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "cluster_ensure",
        "description": (
            "Ensure a ray-manager is running and optionally launch worker "
            "sessions. Resolves the manager from env / persisted state / canfar "
            "discovery, waits for ready, creates workers when requested, and "
            "returns the manager URL, Jobs/Dashboard address, cluster phase and "
            "joined worker count."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Manager connect URL or Jobs API URL.",
                },
                "workers": {
                    "type": "integer",
                    "description": "Worker sessions to launch.",
                    "default": 0,
                },
                "cores": {"type": "integer", "description": "CPUs per worker.", "default": 1},
                "ram": {"type": "integer", "description": "RAM GiB per worker.", "default": 4},
                "gpus": {"type": "integer", "description": "GPUs per worker.", "default": 0},
                "timeout": {
                    "type": "integer",
                    "description": "Wait timeout (seconds).",
                    "default": 1800,
                },
            },
        },
        "handler": _tool_cluster_ensure,
    },
    {
        "name": "cluster_status",
        "description": (
            "Current CANFAR Ray cluster status: phase, ray address, live nodes, "
            "joined/target workers, auth and dashboard path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Manager connect URL or Jobs API URL.",
                },
            },
        },
        "handler": _tool_cluster_status,
    },
    {
        "name": "cluster_scale",
        "description": (
            "Scale worker sessions up or down to a target count "
            "(launches/destroys CANFAR worker sessions)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workers": {"type": "integer", "description": "Target number of worker sessions."},
                "address": {
                    "type": "string",
                    "description": "Manager connect URL or Jobs API URL.",
                },
                "cores": {"type": "integer", "description": "CPUs per new worker.", "default": 1},
                "ram": {"type": "integer", "description": "RAM GiB per new worker.", "default": 4},
                "gpus": {"type": "integer", "description": "GPUs per new worker.", "default": 0},
                "timeout": {
                    "type": "integer",
                    "description": "Wait timeout (seconds).",
                    "default": 1800,
                },
            },
            "required": ["workers"],
        },
        "handler": _tool_cluster_scale,
    },
    {
        "name": "dashboard_url",
        "description": (
            "Resolve the Ray Dashboard / Jobs API URL for the current cluster "
            "(env, persisted state, or canfar discovery)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Manager connect URL or Jobs API URL.",
                },
            },
        },
        "handler": _tool_dashboard_url,
    },
]

_TOOL_BY_NAME = {t["name"]: t for t in TOOLS}


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["inputSchema"],
        }
        for t in TOOLS
    ]


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one decoded JSON-RPC message; return the response (None for notifications)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _json_error(None, INVALID_REQUEST, "Invalid Request")
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params")
    if not isinstance(params, dict):
        params = {}

    if method == "initialize":
        requested = (params.get("protocolVersion") or "").strip()
        version = requested if requested in _SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
        return _json_result(
            msg_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _json_result(msg_id, {})
    if method == "tools/list":
        return _json_result(msg_id, {"tools": _tool_definitions()})
    if method == "tools/call":
        return _handle_tools_call(msg_id, params)
    # JSON-RPC forbids responding to notifications (messages without an id),
    # so unknown notification methods (e.g. notifications/cancelled) get no
    # response — only id-carrying requests get METHOD_NOT_FOUND.
    if msg_id is None:
        return None
    return _json_error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")


def _handle_tools_call(msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    tool = _TOOL_BY_NAME.get(name)
    if tool is None:
        return _json_error(msg_id, INVALID_PARAMS, f"Unknown tool: {name}")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _json_error(msg_id, INVALID_PARAMS, f"Invalid arguments for {name}: expected object")
    try:
        result = tool["handler"](arguments)
    except (TypeError, ValueError) as exc:
        return _json_error(msg_id, INVALID_PARAMS, f"Invalid arguments for {name}: {exc}")
    except RuntimeError as exc:
        # Business failure (no manager, not ready, unreachable) — report as an
        # error result so the agent sees a clean message, not a JSON-RPC error.
        return _json_result(
            msg_id,
            {
                "content": [{"type": "text", "text": f"{exc}"}],
                "isError": True,
            },
        )
    except Exception as exc:  # noqa: BLE001 — keep the server alive; surface to the client
        return _json_result(
            msg_id,
            {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            },
        )
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    return _json_result(
        msg_id,
        {
            "content": [{"type": "text", "text": text}],
        },
    )


# ---------------------------------------------------------------------------
# stdio loop
# ---------------------------------------------------------------------------


def serve_stdio() -> int:
    """Run the MCP server over stdin/stdout until EOF.

    Each line is one JSON-RPC message; notifications get no response. Returns 0
    on clean EOF.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps(_json_error(None, PARSE_ERROR, f"Parse error: {exc}")) + "\n"
            )
            sys.stdout.flush()
            continue
        response = handle_message(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0
