"""Unit tests for the Ray-native CANFAR autoscaler (CanfarNodeProvider)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub ray (only the NodeProvider base is needed) + canfar client.
# The stub mirrors Ray 2.56's public module path: ray.autoscaler.node_provider.
_ray = types.ModuleType("ray")
_ray_autoscaler = types.ModuleType("ray.autoscaler")
_ray_np = types.ModuleType("ray.autoscaler.node_provider")


class _NodeProvider:
    def __init__(self, provider_config, cluster_name):
        self.provider_config = provider_config
        self.cluster_name = cluster_name


_ray_np.NodeProvider = _NodeProvider
_ray_autoscaler.node_provider = _ray_np
_ray.autoscaler = _ray_autoscaler
sys.modules.setdefault("ray", _ray)
sys.modules.setdefault("ray.autoscaler", _ray_autoscaler)
sys.modules.setdefault("ray.autoscaler.node_provider", _ray_np)

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
    # Keys Ray 2.x StandardAutoscaler.reset() hard-requires (KeyError otherwise).
    for required in (
        "head_node_type: ray.head.default",
        "idle_timeout_minutes: 5",
        "auth: {}",
        "head_setup_commands: []",
        "head_start_ray_commands: []",
        "worker_setup_commands: []",
        "worker_start_ray_commands: []",
    ):
        assert required in text, f"missing {required}"
    # memory is bytes (16 GiB = 16 * 1024**3), not MiB.
    assert f"memory: {16 * 1024 * 1024 * 1024}" in text
    assert "memory: 16384" not in text


def test_write_autoscaling_config_idle_timeout_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """idle_timeout_minutes param + RAY_AUTOSCALING_IDLE_TIMEOUT_MINUTES env win
    over the baked 5-minute default."""
    out = tmp_path / "autoscaling.yaml"
    path = write_autoscaling_config(
        path=out,
        cluster_name="c1",
        worker_count=0,
        max_workers=3,
        idle_timeout_minutes=2,
    )
    assert "idle_timeout_minutes: 2" in path.read_text(encoding="utf-8")

    # env fallback when the param is not passed
    monkeypatch.setenv("RAY_AUTOSCALING_IDLE_TIMEOUT_MINUTES", "1")
    path2 = write_autoscaling_config(
        path=tmp_path / "a2.yaml",
        cluster_name="c1",
        worker_count=0,
        max_workers=3,
    )
    assert "idle_timeout_minutes: 1" in path2.read_text(encoding="utf-8")


def test_provider_merges_nested_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray passes the whole provider dict; options live under `config`."""
    provider = CanfarNodeProvider(
        {
            "type": "external",
            "module": "astroai_workload.autoscaler.CanfarNodeProvider",
            "config": {
                "worker_image": "img/ray-worker:t",
                "cores": 4,
                "ram_gb": 16,
                "gpus": 2,
                "ray_version": "2.56.1",
            },
        },
        "c1",
    )
    assert provider._worker_image() == "img/ray-worker:t"
    assert provider._worker_spec() == {"cores": 4, "ram_gb": 16, "gpus": 2}
    assert provider._worker_env()["RAY_CLUSTER_ID"] == "c1"
    assert provider._worker_env()["RAY_WORKER_CPUS"] == "4"
    assert provider._worker_env()["RAY_WORKER_GPUS"] == "2"


def test_import_guard_falls_back_on_missing_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModuleNotFoundError (a subclass of ImportError) keeps the graceful fallback.

    Without ray installed (py3.13 base images) the module must stay importable,
    degrading _RayNodeProvider to object and recording the import error.
    """
    import builtins
    import importlib

    import astroai_workload.autoscaler as autoscaler_mod

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ray.autoscaler.node_provider" or name.startswith(
            "ray.autoscaler.node_provider."
        ):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        reloaded = importlib.reload(autoscaler_mod)
        assert reloaded._RayNodeProvider is object
        assert isinstance(reloaded._RAY_NODE_PROVIDER_IMPORT_ERROR, ModuleNotFoundError)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(autoscaler_mod)  # restore normal module state


def test_import_guard_propagates_non_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine bug inside ray's import chain must fail loudly, not be swallowed."""
    import builtins
    import importlib

    import astroai_workload.autoscaler as autoscaler_mod

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ray.autoscaler.node_provider":
            raise TypeError("boom inside ray import chain")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(TypeError, match="boom inside ray import chain"):
            importlib.reload(autoscaler_mod)
    finally:
        # Failure-safe restore: even if the guard regresses and reload does NOT
        # raise (so pytest.raises fails), never leave the shared module object
        # in the fallback state for later tests in this file.
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(autoscaler_mod)  # restore normal module state
