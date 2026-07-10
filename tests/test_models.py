import json
from datetime import datetime, timezone

import pytest

from astroai_workload import (
    DataProductRef,
    ProvenanceManifest,
    ResourceRequest,
    RunSpec,
    RunStatus,
)


def test_run_status_terminal_states() -> None:
    assert RunStatus.SUCCEEDED.terminal
    assert RunStatus.FAILED.terminal
    assert RunStatus.STOPPED.terminal
    assert not RunStatus.RUNNING.terminal


def test_resource_request_validates_and_freezes_custom_resources() -> None:
    custom = {"accelerator_type:A100": 1.0}
    request = ResourceRequest(cpus=4, custom=custom)
    custom["accelerator_type:A100"] = 2.0
    assert request.custom["accelerator_type:A100"] == 1.0
    with pytest.raises(ValueError):
        ResourceRequest(cpus=-1)
    with pytest.raises(ValueError):
        ResourceRequest(memory_bytes=0)


def test_run_spec_requires_identity_and_command() -> None:
    with pytest.raises(ValueError):
        RunSpec(run_id="", command=("true",))
    with pytest.raises(ValueError):
        RunSpec(run_id="run", command=())


def test_run_spec_is_json_serializable() -> None:
    spec = RunSpec(
        run_id="run",
        command=("python", "driver.py"),
        inputs=(DataProductRef("vos://input.fits"),),
        metadata={"seed": 4},
    )
    assert json.loads(json.dumps(spec.to_dict()))["command"] == ["python", "driver.py"]


def test_provenance_manifest_is_json_compatible() -> None:
    product = DataProductRef("vos://bucket/catalog.parquet", size_bytes=12)
    manifest = ProvenanceManifest(
        run_id="run-1",
        inputs=(product,),
        parameters={"seed": 7},
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    payload = manifest.to_dict()
    assert payload["created_at"] == "2026-01-02T00:00:00+00:00"
    assert payload["inputs"][0]["uri"] == product.uri
    assert payload["schema_version"] == "1.0"
