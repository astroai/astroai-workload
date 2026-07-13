#!/usr/bin/env python3
"""Submit train.py to the local ray-manager Jobs API (no host guessing)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from astroai_workload import RayExecutor, ResourceRequest, RunSpec, RunStatus

HERE = Path(__file__).resolve().parent


def main() -> None:
    run_id = f"mnist-{uuid.uuid4().hex[:8]}"
    spec = RunSpec(
        run_id=run_id,
        command=("python", "train.py", "--epochs", "1", "--ckpt", "mnist.pt"),
        resources=ResourceRequest(cpus=1, memory="2GiB"),
        working_directory=str(HERE),
    )
    ex = RayExecutor()  # ASTROAI_RAY_JOBS_ADDRESS on ray-manager
    print(f"submitting {run_id} …")
    ex.submit(spec)
    while True:
        status = ex.status(run_id)
        print("status:", status.value)
        if status.terminal:
            break
        time.sleep(2)
    print(ex.logs(run_id))
    if status is not RunStatus.SUCCEEDED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
