"""Unit tests for the Ray-native CANFAR autoscaler (CanfarNodeProvider)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub ray (only the NodeProvider base is needed) + canfar client.
_ray = types.ModuleType("ray")
_ray_autoscaler = types.ModuleType("ray.autoscaler")
_ray_private = types.ModuleType("ray.autoscaler._private")
_ray_np = types.ModuleType("ray.autoscaler._private.node_provider")


class _NodeProvider:
    def __init__(self, provider_config, cluster_name):
        self.provider_config = provider_config
        self.cluster_name = cluster_name


_ray_np.NodeProvider = _NodeProvider
_ray_private.node_provider = _ray_np
_ray_autoscaler._private = _ray_private
_ray.autoscaler = _ray_autoscaler
sys.modules.setdefault("ray", _ray)
sys.modules.setdefault("ray.autoscaler", _ray_autoscaler)
sys.modules.setdefault("ray.autoscaler._private", _ray_private)
sys.modules.setdefault("ray.autoscaler._private.node_provider", _ray_np)

_canfar = types.ModuleType("canfar")
_canfar_models = types.ModuleType("canfar.models")
_canfar_config = types.ModuleType("canfar.models.config")
_canfar_sessions = types.ModuleType("canfar.sessions")


class _Configuration:
    def __init__(self) -> None:
        self.active = MagicMock(authentication=None, server=None)
        self.registry = MagicMock(username=None, secret=None, url=None)

    def get_credential(self, _idp: str) -> None:
        raise KeyError(_idp)


class _Session:
    def __init__(self) -> None:
        self.config = MagicMock(registry=MagicMock(username=None, secret=None))

    def fetch(self, **_kwargs):
        return []

    def create(self, **_kwargs):
        return []

    def info(self, *_a, **_k):
        return []

    def logs(self, *_a, **_k):
        return {}

    def destroy(self, *_a, **_k):
        return {}


_canfar_config.Configuration = _Configuration
_canfar_sessions.Session = _Session
sys.modules.setdefault("canfar", _canfar)
sys.modules.setdefault("canfar.models", _canfar_models)
sys.modules.setdefault("canfar.models.config", _canfar_config)
sys.modules.setdefault("canfar.sessions", _canfar_sessions)

from astroai_workload.autoscaler import (  # noqa: E402
    CanfarNodeProvider,
    write_autoscaling_config,
)


def _provider(**config):
    cfg = {
        "worker_image": "images.canfar.net/astroai/ray-worker:local",
        "cores": 2,
        "ram_gb": 8,
        "gpus": 1,
        "heartbeat_path": "/arc/home/u/.astroai/ray/clusters/c1/manager-heartbeat",
        "ray_version": "2.56.1",
        **config,
    }
    return CanfarNodeProvider(cfg, "c1")


def test_create_node_launches_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    launches = [
        MagicMock(session_id="sid-1", name="ray-w-c1-1"),
        MagicMock(session_id="sid-2", name="ray-w-c1-2"),
    ]
    provider._ops.create_headless = MagicMock(return_value=launches)

    result = provider.create_node(
        node_config={}, tags={"ray_node_type_name": "ray.worker.default"}, count=2
    )

    assert set(result) == {"sid-1", "sid-2"}
    assert all(ip == "" for ip in result.values())  # IP resolved lazily
    provider._ops.create_headless.assert_called_once()
    kwargs = provider._ops.create_headless.call_args.kwargs
    assert kwargs["replicas"] == 2
    assert kwargs["cores"] == 2
    assert kwargs["gpu"] == 1
    assert kwargs["env"]["RAY_CLUSTER_ID"] == "c1"
    assert kwargs["env"]["RAY_WORKER_CPUS"] == "2"
    # Distinct `ray-as-` prefix keeps autoscaler nodes out of manager GC scope.
    assert kwargs["name"].startswith("ray-as-c1-")
    # Tags are stored for Ray.
    assert provider.node_tags("sid-1") == {"ray_node_type_name": "ray.worker.default"}


def test_internal_ip_resolves_after_running(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    provider._ops.session_info = MagicMock(return_value={"status": "Pending"})
    assert provider.internal_ip("sid-1") == ""

    provider._ops.session_info = MagicMock(return_value={"status": "Running", "podIP": "10.0.0.9"})
    assert provider.internal_ip("sid-1") == "10.0.0.9"

    provider._ops.session_info = MagicMock(return_value={"status": "Running"})
    provider._ops.session_logs = MagicMock(return_value="Worker 10.0.0.5 joining 10.0.0.1:6379\n")
    assert provider.internal_ip("sid-1") == "10.0.0.5"


def test_terminate_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    provider._ops.destroy = MagicMock(return_value=True)
    provider._tags["sid-1"] = {"ray_node_type_name": "ray.worker.default"}
    provider.terminate_node("sid-1")
    provider._ops.destroy.assert_called_once_with("sid-1")
    assert provider.node_tags("sid-1") == {}

    provider._ops.session_status = MagicMock(return_value="Running")
    assert provider.is_running("sid-1") is True
    assert provider.is_terminated("sid-1") is False

    provider._ops.session_status = MagicMock(return_value="Failed")
    assert provider.is_terminated("sid-1") is True


def test_non_terminated_nodes_filters_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    provider._ops.list_headless_sessions = MagicMock(
        return_value=[
            {"id": "a", "status": "Running"},
            {"id": "b", "status": "Pending"},
            {"id": "c", "status": "Succeeded"},
        ]
    )
    nodes = provider.non_terminated_nodes(tag_filters={})
    assert set(nodes) == {"a", "b"}
    # Queries the autoscaler's own session namespace, not manager workers.
    provider._ops.list_headless_sessions.assert_called_once_with(name_prefix="ray-as-c1")


def test_write_autoscaling_config(tmp_path: pytest.MonkeyPatch) -> None:
    out = tmp_path / "autoscaling.yaml"
    path = write_autoscaling_config(
        path=out,
        cluster_name="c1",
        worker_count=2,
        max_workers=8,
        cores=4,
        ram_gb=16,
        gpus=0,
        worker_image="img/ray-worker:t",
    )
    text = path.read_text(encoding="utf-8")
    assert "cluster_name: c1" in text
    assert "max_workers: 8" in text
    assert "module: astroai_workload.autoscaler.CanfarNodeProvider" in text
    assert "min_workers: 2" in text
    assert "max_workers: 8" in text
    assert '"cores": 4' in text
    assert "astroai_workload.autoscaler.CanfarNodeProvider" in text
