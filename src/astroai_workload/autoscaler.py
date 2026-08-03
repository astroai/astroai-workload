"""Ray-native autoscaler backed by CANFAR headless sessions.

Implements :class:`ray.autoscaler.node_provider.NodeProvider` (public path in
Ray 2.56+; the old ``_private.node_provider`` module no longer exists) so
Ray's own autoscaler (started on the head with ``--autoscaling-config``) can
launch and terminate Skaha ``ray-worker`` sessions on demand. Node IDs are
CANFAR session IDs.

The provider is used from inside the ray-manager head pod, so it requires the
``astroai-workload[cluster]`` extra (canfar client). It does **not** need Ray's
autoscaler internals beyond the NodeProvider base class; all session work goes
through :class:`astroai_workload.canfar_ops.CanfarOps`.

Provider config (``provider.config`` in the autoscaling YAML)::

    worker_image: images.canfar.net/astroai/ray-worker:<tag>
    cores: 1
    ram_gb: 4
    gpus: 0
    heartbeat_path: /arc/home/<user>/.astroai/ray/clusters/<id>/manager-heartbeat
    cluster_id: default
    node_manager_port, object_manager_port, ...  (optional Ray port pins)

Ray polls :meth:`internal_ip` until the worker is Running and its pod IP is
known (parsed from the session log's ``Worker <ip> joining`` line), so
asynchronous Skaha startup is handled natively.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from astroai_workload.canfar_ops import CanfarOps
from astroai_workload.ray_cluster import parse_worker_ip_from_logs
from astroai_workload.settings import manager_pod_ip

logger = logging.getLogger(__name__)

# Ray 2.56 moved the provider base to the public `ray.autoscaler.node_provider`
# module (the old `_private.node_provider` path no longer exists). Keep the
# module importable without ray (py3.13 images) but never swallow the failure
# silently — record it so the RuntimeError below names the real cause. Catch
# only ImportError (which covers ModuleNotFoundError for a missing ray): a
# genuine bug elsewhere in ray's import chain must fail loudly, not silently
# degrade the provider to an object subclass.
_RAY_NODE_PROVIDER_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - exercised in images where ray is installed
    from ray.autoscaler.node_provider import NodeProvider as _RayNodeProvider
except ImportError as exc:
    _RayNodeProvider = object  # type: ignore[assignment,misc]
    _RAY_NODE_PROVIDER_IMPORT_ERROR = exc
    logger.warning("Ray NodeProvider import failed; autoscaler disabled: %r", exc)

_TERMINAL_SESSION_STATUSES = {"Failed", "Error", "Succeeded", "Completed", "Terminating"}
_RUNNING_SESSION_STATUSES = {"Running"}


def _default_worker_image() -> str:
    tag = os.environ.get("RAY_IMAGE_TAG", os.environ.get("BUILD_TAG", "latest"))
    registry = os.environ.get("REGISTRY", "images.canfar.net")
    owner = os.environ.get("OWNER", "astroai")
    return f"{registry}/{owner}/ray-worker:{tag}"


class CanfarNodeProvider(_RayNodeProvider):  # type: ignore[misc,valid-type]
    """Ray autoscaler node provider that creates CANFAR headless sessions."""

    def __init__(self, provider_config: dict[str, Any], cluster_name: str) -> None:
        if _RayNodeProvider is object:  # pragma: no cover - guard for py3.13-only envs
            raise RuntimeError(
                "Ray autoscaler support requires astroai-workload[jobs] (ray installed). "
                f"NodeProvider import failed: {_RAY_NODE_PROVIDER_IMPORT_ERROR!r}"
            )
        super().__init__(provider_config, cluster_name)
        self.provider_config = provider_config or {}
        # Ray constructs external providers with the whole provider dict
        # (type/module/config). Merge the nested ``config`` so the accessors
        # below (which read top-level keys) see the written options either way.
        nested = self.provider_config.get("config")
        if isinstance(nested, dict):
            self.provider_config = {**self.provider_config, **nested}
        self.cluster_name = cluster_name
        self._ops = CanfarOps()
        self._tags: dict[str, dict[str, str]] = {}

    # -- config helpers -----------------------------------------------------

    def _worker_image(self) -> str:
        return str(self.provider_config.get("worker_image") or _default_worker_image())

    def _worker_spec(self) -> dict[str, Any]:
        return {
            "cores": int(self.provider_config.get("cores", 1)),
            "ram_gb": int(self.provider_config.get("ram_gb", 4)),
            "gpus": int(self.provider_config.get("gpus", 0)),
        }

    def _heartbeat_path(self) -> str:
        return str(
            self.provider_config.get("heartbeat_path")
            or os.environ.get("RAY_MANAGER_HEARTBEAT_PATH", "")
        )

    def _worker_env(self) -> dict[str, str]:
        spec = self._worker_spec()
        env: dict[str, str] = {
            "RAY_CLUSTER_ID": self.cluster_name,
            "RAY_HEAD_IP": manager_pod_ip(),
            "RAY_HEAD_PORT": str(
                self.provider_config.get("ray_head_port") or os.environ.get("RAY_HEAD_PORT", "6379")
            ),
            "RAY_VERSION_EXPECTED": str(
                self.provider_config.get("ray_version")
                or os.environ.get("RAY_VERSION_EXPECTED", "")
            ),
            "RAY_WORKER_CPUS": str(spec["cores"]),
            "RAY_WORKER_GPUS": str(spec["gpus"]),
            "RAY_SPILL_DIR": str(
                self.provider_config.get("spill_dir")
                or f"{os.environ.get('SCRATCH', '/scratch')}/ray/{self.cluster_name}"
            ),
        }
        hb = self._heartbeat_path()
        if hb:
            env["RAY_MANAGER_HEARTBEAT_PATH"] = hb
            env["RAY_MANAGER_HEARTBEAT_TIMEOUT_SECONDS"] = str(
                self.provider_config.get("heartbeat_timeout_seconds")
                or os.environ.get("RAY_MANAGER_HEARTBEAT_TIMEOUT_SECONDS", "120")
            )
        for key in (
            "RAY_NODE_MANAGER_PORT",
            "RAY_OBJECT_MANAGER_PORT",
            "RAY_RUNTIME_ENV_AGENT_PORT",
            "RAY_DASHBOARD_AGENT_GRPC_PORT",
            "RAY_MIN_WORKER_PORT",
            "RAY_MAX_WORKER_PORT",
        ):
            val = self.provider_config.get(key) or os.environ.get(key)
            if val:
                env[key] = str(val)
        return env

    # -- NodeProvider contract ---------------------------------------------

    def create_node(
        self, node_config: dict[str, Any], tags: dict[str, str], count: int
    ) -> dict[str, str]:
        """Launch *count* headless ray-worker sessions; return {node_id: ip}."""
        spec = self._worker_spec()
        # Distinct ``ray-as-`` prefix (autoscaler-managed) so the manager's
        # orphan GC (which owns ``ray-w-``/``ray-retry-``/``ray-preflight-``)
        # never destroys autoscaler nodes, and Ray's autoscaler never adopts
        # manager-created workers (non_terminated_nodes matches this prefix).
        name = f"ray-as-{self.cluster_name}-{int(time.time() * 1000)}"[:60]
        launches = self._ops.create_headless(
            name=name,
            image=self._worker_image(),
            cores=spec["cores"],
            ram=spec["ram_gb"],
            gpu=spec["gpus"] or None,
            env=self._worker_env(),
            replicas=count,
        )
        result: dict[str, str] = {}
        for launch in launches:
            self._tags[launch.session_id] = dict(tags or {})
            # Ray resolves IPs lazily via internal_ip(); session may still be Pending.
            result[launch.session_id] = ""
        return result

    def terminate_node(self, node_id: str) -> None:
        self._tags.pop(node_id, None)
        self._ops.destroy(node_id)

    def terminate_nodes(self, node_ids: list[str]) -> None:
        for node_id in node_ids:
            self.terminate_node(node_id)

    def non_terminated_nodes(self, tag_filters: dict[str, str]) -> list[str]:
        out: list[str] = []
        for row in self._ops.list_headless_sessions(name_prefix=f"ray-as-{self.cluster_name}"):
            sid = str(row.get("id") or "")
            if not sid:
                continue
            status = str(row.get("status") or "Unknown")
            if status in _TERMINAL_SESSION_STATUSES:
                continue
            out.append(sid)
        return out

    def is_running(self, node_id: str) -> bool:
        return self._ops.session_status(node_id) in _RUNNING_SESSION_STATUSES

    def is_terminated(self, node_id: str) -> bool:
        return self._ops.session_status(node_id) in _TERMINAL_SESSION_STATUSES

    def node_tags(self, node_id: str) -> dict[str, str]:
        return dict(self._tags.get(node_id) or {})

    def set_node_tags(self, node_id: str, tags: dict[str, str]) -> None:
        self._tags[node_id] = dict(tags or {})

    def internal_ip(self, node_id: str) -> str:
        info = self._ops.session_info(node_id)
        status = str(info.get("status") or "Unknown")
        if status not in _RUNNING_SESSION_STATUSES:
            return ""
        # Prefer an explicit IP field, else parse the worker log's join line.
        ip = str(info.get("podIP") or info.get("nodeIP") or info.get("ip") or "").strip()
        if ip:
            return ip
        logs = self._ops.session_logs(node_id)
        parsed = parse_worker_ip_from_logs(logs)
        return parsed or ""

    def external_ip(self, node_id: str) -> str:
        return self.internal_ip(node_id)


def write_autoscaling_config(
    *,
    path: str | Path,
    cluster_name: str,
    worker_count: int,
    max_workers: int,
    cores: int = 1,
    ram_gb: int = 4,
    gpus: int = 0,
    worker_image: str | None = None,
    ray_version: str | None = None,
    ray_head_port: int = 6379,
    heartbeat_path: str | None = None,
    spill_dir: str | None = None,
    idle_timeout_minutes: int | None = None,
) -> Path:
    """Write a Ray autoscaling YAML using the CANFAR node provider.

    ``worker_count`` is the initial ``min_workers``; the autoscaler grows to
    ``max_workers`` on demand and shrinks back when idle.

    ``idle_timeout_minutes`` overrides the baked 5-minute idle timeout
    (default: env ``RAY_AUTOSCALING_IDLE_TIMEOUT_MINUTES`` or 5) — tests and
    operators can shrink idle workers faster without rebuilding.
    """
    from astroai_workload.settings import _default_ray_version

    version = ray_version or os.environ.get("RAY_VERSION_EXPECTED", "") or _default_ray_version()
    image = worker_image or _default_worker_image()
    hb = heartbeat_path or os.environ.get("RAY_MANAGER_HEARTBEAT_PATH", "")
    spill = spill_dir or f"{os.environ.get('SCRATCH', '/scratch')}/ray/{cluster_name}"
    idle = idle_timeout_minutes if idle_timeout_minutes is not None else int(
        os.environ.get("RAY_AUTOSCALING_IDLE_TIMEOUT_MINUTES", "5")
    )

    provider_config: dict[str, Any] = {
        "worker_image": image,
        "cores": cores,
        "ram_gb": ram_gb,
        "gpus": gpus,
        "ray_head_port": ray_head_port,
        "ray_version": version,
        "spill_dir": spill,
    }
    if hb:
        provider_config["heartbeat_path"] = hb
        provider_config["heartbeat_timeout_seconds"] = int(
            os.environ.get("RAY_MANAGER_HEARTBEAT_TIMEOUT_SECONDS", "120")
        )

    # Ray 2.x autoscaler requires these keys; workers join the cluster natively
    # (start-worker.sh in the worker image), so the command lists stay empty.
    # ``head_node_type``/``auth``/``idle_timeout_minutes`` are mandatory reads
    # in StandardAutoscaler.reset() — omitting them crashes the monitor with
    # KeyError the moment `ray start --head --autoscaling-config` boots.
    yaml = f"""\
# Generated by astroai-workload ({__name__}); do not edit by hand.
cluster_name: {cluster_name}
max_workers: {max_workers}
head_node_type: ray.head.default
idle_timeout_minutes: {idle}

auth: {{}}

provider:
  type: external
  module: astroai_workload.autoscaler.CanfarNodeProvider
  config: {_yaml_inline(provider_config)}

available_node_types:
  ray.head.default:
    resources:
      CPU: 0
      memory: 0
    node_config: {{}}
  ray.worker.default:
    min_workers: {worker_count}
    max_workers: {max_workers}
    resources:
      CPU: {cores}
      memory: {ram_gb * 1024 * 1024 * 1024}
    node_config: {{}}

file_mounts: {{}}
cluster_synced_files: []
env_vars: {{}}
head_setup_commands: []
head_start_ray_commands: []
worker_setup_commands: []
worker_start_ray_commands: []
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml, encoding="utf-8")
    return path


def _yaml_inline(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, sort_keys=True)
