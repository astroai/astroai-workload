"""Ray Jobs implementation of the compute-facility boundary."""

from __future__ import annotations

import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from .models import ResourceRequest, RunSpec, RunStatus

_DEFAULT_JOBS_ADDRESS = "http://127.0.0.1:8265"
_ADDRESS_HINT = (
    "Ray Jobs address not reachable. On CANFAR: launch a contributed ray-manager "
    "session, open the connectURL from `canfar ps`, create workers, then run "
    "RayExecutor() on that manager (ASTROAI_RAY_JOBS_ADDRESS is set there). "
    "Do not invent hostnames like ray-manager:8265."
)
_RAY_MISSING_HINT = (
    "Ray is not installed in this Python. Install the astroai-workload[jobs] "
    "extra (or run inside a ray-manager / ray-worker image venv) to use the Ray "
    "Jobs client. Cluster lifecycle (`astroai-workload cluster ...`) does not "
    "need Ray."
)


def _job_submission_client_class() -> type[Any]:
    """Import the Ray Jobs client lazily; raise a clear error when Ray is absent."""
    try:
        from ray.job_submission import JobSubmissionClient
    except ImportError as exc:  # noqa: BLE001 — any failure to import ray is actionable
        raise RuntimeError(_RAY_MISSING_HINT) from exc
    return JobSubmissionClient


def resolve_jobs_address(address: str | None = None) -> str:
    """Resolve the Ray Jobs / Dashboard URL without guessing cluster DNS."""

    if address and address.strip():
        return address.strip()
    for key in ("ASTROAI_RAY_JOBS_ADDRESS", "RAY_DASHBOARD_URL"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return _DEFAULT_JOBS_ADDRESS


class RayExecutor:
    """Submit driver commands through the Ray Jobs API.

    Cluster creation and worker lifecycle stay in :mod:`astroai_workload.cluster`
    (CANFAR ray-manager + workers). When ``address`` is omitted, uses
    ``ASTROAI_RAY_JOBS_ADDRESS`` / ``RAY_DASHBOARD_URL``, else localhost:8265
    (correct inside the ray-manager pod).
    """

    def __init__(self, address: str | None = None, *, client: Any | None = None) -> None:
        if client is None:
            resolved = resolve_jobs_address(address)
            cls = _job_submission_client_class()
            try:
                client = cls(resolved)
            except Exception as exc:  # noqa: BLE001 — surface actionable hint
                raise ConnectionError(f"{_ADDRESS_HINT} (tried {resolved!r})") from exc
        self._client = client

    def submit(self, spec: RunSpec) -> str:
        runtime_env: dict[str, Any] = {"env_vars": dict(spec.environment)}
        if spec.working_directory is not None:
            runtime_env["working_dir"] = spec.working_directory
        metadata = {"astroai_run_id": spec.run_id}
        metadata.update({str(key): str(value) for key, value in spec.metadata.items()})
        metadata["astroai_contract"] = "astroai-workload.v1"
        metadata["astroai_resources"] = json.dumps(
            spec.resources.to_dict(), sort_keys=True, separators=(",", ":")
        )
        metadata["astroai_inputs"] = json.dumps(
            [product.to_dict() for product in spec.inputs],
            sort_keys=True,
            separators=(",", ":"),
        )
        metadata["astroai_expected_outputs"] = json.dumps(
            [product.to_dict() for product in spec.expected_outputs],
            sort_keys=True,
            separators=(",", ":"),
        )
        if spec.resources.memory_bytes is not None:
            metadata["astroai_memory_bytes"] = str(spec.resources.memory_bytes)
        if spec.resources.walltime_seconds is not None:
            metadata["astroai_walltime_seconds"] = str(spec.resources.walltime_seconds)
        submit_kwargs: dict[str, Any] = {
            "entrypoint": shlex.join(spec.command),
            "submission_id": spec.run_id,
            "runtime_env": runtime_env,
            "metadata": metadata,
            "entrypoint_num_cpus": spec.resources.cpus,
            "entrypoint_num_gpus": spec.resources.gpus,
            "entrypoint_resources": dict(spec.resources.custom),
        }
        # Honor memory as a real Ray Jobs entrypoint reservation (not metadata-only).
        if spec.resources.memory_bytes is not None:
            submit_kwargs["entrypoint_memory"] = spec.resources.memory_bytes
        return str(self._client.submit_job(**submit_kwargs))

    def status(self, run_id: str) -> RunStatus:
        raw = self._client.get_job_status(run_id)
        value = str(getattr(raw, "value", raw)).lower()
        return {
            "pending": RunStatus.PENDING,
            "running": RunStatus.RUNNING,
            "succeeded": RunStatus.SUCCEEDED,
            "completed": RunStatus.SUCCEEDED,
            "failed": RunStatus.FAILED,
            "stopped": RunStatus.STOPPED,
            "stopping": RunStatus.STOPPED,
            "cancelled": RunStatus.STOPPED,
            "canceled": RunStatus.STOPPED,
        }.get(value, RunStatus.UNKNOWN)

    def cancel(self, run_id: str) -> None:
        self._client.stop_job(run_id)

    def logs(self, run_id: str) -> str:
        return str(self._client.get_job_logs(run_id))

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return Jobs known to the Dashboard API (newest-friendly summary rows)."""
        raw = self._client.list_jobs()
        rows: list[dict[str, Any]] = []
        for job in raw or ():
            if isinstance(job, dict):
                rows.append(dict(job))
                continue
            status = getattr(job, "status", None)
            status_value = getattr(status, "value", status)
            rows.append(
                {
                    "submission_id": getattr(job, "submission_id", None)
                    or getattr(job, "job_id", None),
                    "job_id": getattr(job, "job_id", None),
                    "status": str(status_value).lower() if status_value is not None else None,
                    "entrypoint": getattr(job, "entrypoint", None),
                    "start_time": getattr(job, "start_time", None),
                    "end_time": getattr(job, "end_time", None),
                    "metadata": dict(getattr(job, "metadata", None) or {}),
                }
            )
        return rows

    def wait(
        self,
        run_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> tuple[RunStatus, str]:
        """Block until *run_id* reaches a terminal status.

        Returns ``(status, logs)``.  When *timeout* expires the current
        status is returned (may be ``UNKNOWN``).
        """
        start = time.monotonic()
        while True:
            status = self.status(run_id)
            if status.terminal:
                return status, self.logs(run_id)
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return RunStatus.UNKNOWN, self.logs(run_id)
            time.sleep(poll_interval)

    def submit_and_wait(
        self,
        spec: RunSpec,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> tuple[RunStatus, str]:
        """Submit a run and block until it finishes.  Returns ``(status, logs)``."""
        self.submit(spec)
        return self.wait(spec.run_id, poll_interval=poll_interval, timeout=timeout)


def run_script(
    script: str | Path,
    *,
    address: str | None = None,
    cpus: float = 1,
    memory: str | None = None,
    gpus: float = 0,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    working_directory: str | None = None,
    run_id: str | None = None,
) -> tuple[RunStatus, str]:
    """Run a Python script as a Ray Job and wait for completion.

    Zero Ray knowledge required — just point at a script.

    ``memory`` (e.g. ``"4GiB"``) reserves Ray entrypoint memory when set.
    Leave it ``None`` unless the cluster has enough free memory to schedule.

    Returns ``(status, logs)``.

    Example::

        from astroai_workload import run_script, RunStatus
        status, logs = run_script("train.py", cpus=2, memory="8GiB")
        if status != RunStatus.SUCCEEDED:
            print(logs)
    """
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = ["python", str(script_path)]
    if args:
        command.extend(args)

    resources = ResourceRequest(cpus=cpus, gpus=gpus)
    if memory is not None:
        resources = ResourceRequest(cpus=cpus, gpus=gpus, memory=memory)

    spec = RunSpec(
        run_id=run_id or f"run-{uuid.uuid4().hex[:8]}",
        command=tuple(command),
        resources=resources,
        environment=env or {},
        working_directory=working_directory or str(script_path.parent.resolve()),
    )

    ex = RayExecutor(address=address)
    return ex.submit_and_wait(spec, timeout=timeout)
