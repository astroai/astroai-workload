#!/usr/bin/env python3
"""Submit examples/smoke/job.py via astroai-workload (no torch)."""

from __future__ import annotations

from pathlib import Path

from astroai_workload import run_script, RunStatus

HERE = Path(__file__).resolve().parent


def main() -> None:
    status, logs = run_script(
        HERE / "job.py",
        cpus=1,
        memory="1GiB",
        run_id=None,
    )
    print(logs)
    if status is not RunStatus.SUCCEEDED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
