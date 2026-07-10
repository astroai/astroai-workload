"""Optional Ray Jobs implementation of the compute-facility boundary."""

from __future__ import annotations

import shlex
from typing import Any

from .models import RunSpec, RunStatus


class RayExecutor:
    """Submit driver commands through the Ray Jobs API.

    Ray is imported lazily so the core contracts remain dependency-free.
    Cluster creation and worker lifecycle are deliberately outside this class.
    """

    def __init__(self, address: str | None = None, *, client: Any | None = None) -> None:
        if client is None:
            try:
                from ray.job_submission import JobSubmissionClient
            except ImportError as exc:  # pragma: no cover - depends on optional install
                raise ImportError(
                    "RayExecutor requires the 'ray' extra: pip install 'astroai-workload[ray]'"
                ) from exc
            client = JobSubmissionClient(address)
        self._client = client

    def submit(self, spec: RunSpec) -> str:
        runtime_env: dict[str, Any] = {"env_vars": dict(spec.environment)}
        if spec.working_directory is not None:
            runtime_env["working_dir"] = spec.working_directory
        metadata = {"astroai_run_id": spec.run_id}
        metadata.update({str(key): str(value) for key, value in spec.metadata.items()})
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
            "failed": RunStatus.FAILED,
            "stopped": RunStatus.STOPPED,
        }.get(value, RunStatus.UNKNOWN)

    def cancel(self, run_id: str) -> None:
        self._client.stop_job(run_id)

    def logs(self, run_id: str) -> str:
        return str(self._client.get_job_logs(run_id))
