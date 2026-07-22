"""Ray Jobs implementation of the compute-facility boundary."""

from __future__ import annotations

import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from ray.job_submission import JobSubmissionClient

from .models import ResourceRequest, RunSpec, RunStatus

_DEFAULT_JOBS_ADDRESS = "http://127.0.0.1:8265"
_ADDRESS_HINT = (
    "Ray Jobs address not reachable. On CANFAR: launch a contributed ray-manager "
    "session, open the connectURL from `canfar ps`, create workers, then run "
    "RayExecutor() on that manager (ASTROAI_RAY_JOBS_ADDRESS is set there). "
    "Do not invent hostnames like ray-manager:8265."
)


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

    Cluster creation and worker lifecycle stay outside this class (CANFAR
    ray-manager + workers). When ``address`` is omitted, uses
    ``ASTROAI_RAY_JOBS_ADDRESS`` / ``RAY_DASHBOARD_URL``, else localhost:8265
    (correct inside the ray-manager pod).
    """

    def __init__(self, address: str | None = None, *, client: Any | None = None) -> None:
        if client is None:
            resolved = resolve_jobs_address(address)
            try:
                client = JobSubmissionClient(resolved)
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
        return str(
            self._client.submit_job(
                entrypoint=shlex.join(spec.command),
                submission_id=spec.run_id,
                runtime_env=runtime_env,
                metadata=metadata,
                entrypoint_num_cpus=spec.resources.cpus,
                entrypoint_num_gpus=spec.resources.gpus,
                entrypoint_resources=dict(spec.resources.custom),
            )
        )

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
    memory: str = "4GiB",
    gpus: float = 0,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    working_directory: str | None = None,
    run_id: str | None = None,
) -> tuple[RunStatus, str]:
    """Run a Python script as a Ray Job and wait for completion.

    Zero Ray knowledge required — just point at a script.

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

    spec = RunSpec(
        run_id=run_id or f"run-{uuid.uuid4().hex[:8]}",
        command=tuple(command),
        resources=ResourceRequest(cpus=cpus, memory=memory, gpus=gpus),
        environment=env or {},
        working_directory=working_directory or str(script_path.parent.resolve()),
    )

    ex = RayExecutor(address=address)
    return ex.submit_and_wait(spec, timeout=timeout)
