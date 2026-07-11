import json
from enum import Enum
from types import SimpleNamespace

from astroai_workload import RayExecutor, ResourceRequest, RunSpec, RunStatus


class _JobState(Enum):
    RUNNING = "RUNNING"


class FakeRayClient:
    def __init__(self) -> None:
        self.submission = None
        self.stopped = None

    def submit_job(self, **kwargs):
        self.submission = kwargs
        return kwargs["submission_id"]

    def get_job_status(self, run_id):
        return _JobState.RUNNING

    def stop_job(self, run_id):
        self.stopped = run_id

    def get_job_logs(self, run_id):
        return f"logs:{run_id}"


def test_ray_executor_adapts_run_spec_without_managing_cluster() -> None:
    client = FakeRayClient()
    executor = RayExecutor(client=client)
    spec = RunSpec(
        run_id="photoz-12",
        command=("python", "fit model.py", "--seed", "7"),
        resources=ResourceRequest(
            cpus=4,
            gpus=1,
            memory_bytes=8_000_000_000,
            walltime_seconds=3600,
            custom={"node_type": 1},
        ),
        environment={"OMP_NUM_THREADS": "4"},
        working_directory="vos://code/release.zip",
        metadata={"campaign": "deep"},
    )

    assert executor.submit(spec) == spec.run_id
    assert client.submission["entrypoint"] == "python 'fit model.py' --seed 7"
    assert client.submission["entrypoint_num_cpus"] == 4
    assert client.submission["entrypoint_num_gpus"] == 1
    assert client.submission["metadata"]["astroai_memory_bytes"] == "8000000000"
    assert client.submission["metadata"]["astroai_walltime_seconds"] == "3600"
    assert client.submission["metadata"]["astroai_contract"] == "astroai-workload.v1"
    assert json.loads(client.submission["metadata"]["astroai_resources"])["gpus"] == 1
    assert json.loads(client.submission["metadata"]["astroai_inputs"]) == []
    assert json.loads(client.submission["metadata"]["astroai_expected_outputs"]) == []
    assert client.submission["runtime_env"]["env_vars"] == {"OMP_NUM_THREADS": "4"}
    assert executor.status(spec.run_id) is RunStatus.RUNNING
    assert executor.logs(spec.run_id) == "logs:photoz-12"
    executor.cancel(spec.run_id)
    assert client.stopped == spec.run_id


def test_ray_executor_normalizes_terminal_status_variants() -> None:
    client = FakeRayClient()
    executor = RayExecutor(client=client)
    for raw, expected in (("COMPLETED", RunStatus.SUCCEEDED), ("STOPPING", RunStatus.STOPPED)):
        client.get_job_status = lambda _run_id, raw=raw: SimpleNamespace(value=raw)
        assert executor.status("run") is expected
