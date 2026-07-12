"""Bounded Ray contract benchmark for CANFAR promotion evidence.

The script consumes an existing Ray cluster. Use ``--local`` only for a small
developer smoke; promotion runs should use ``--address auto`` on CANFAR with
``/scratch`` as the temporary root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def benchmark_payload(size: int = 1_048_576) -> bytes:
    """Return a deterministic serializable payload of approximately *size* bytes."""
    if size < 1:
        raise ValueError("size must be positive")
    return (b"astroai-workload\0" * ((size // 16) + 1))[:size]


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run_benchmark(
    *, address: str | None = None, local: bool = False, scratch_root: Path | None = None
) -> dict[str, Any]:
    """Run serialization, retry, cancellation, and cleanup checks on Ray."""
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("install astroai-workload[ray] to run this benchmark") from exc

    if ray.is_initialized():
        owns_ray = False
    else:
        init_kwargs: dict[str, Any] = {"ignore_reinit_error": True, "logging_level": "ERROR"}
        if local:
            init_kwargs.update({"num_cpus": 2, "include_dashboard": False})
        else:
            init_kwargs["address"] = address or os.environ.get("RAY_ADDRESS", "auto")
        ray.init(**init_kwargs)
        owns_ray = True

    if float(ray.cluster_resources().get("CPU", 0.0)) < 2.0:
        if owns_ray:
            ray.shutdown()
        raise RuntimeError("benchmark requires at least two Ray CPUs")

    temporary = scratch_root is None
    if temporary:
        temp_context = tempfile.TemporaryDirectory(prefix="astroai-workload-benchmark-")
        root = Path(temp_context.name)
    else:
        temp_context = None
        root = scratch_root.expanduser().resolve()
        if "astroai" not in root.name.lower():
            raise ValueError("scratch_root must be a dedicated AstroAI benchmark directory")
        root.mkdir(parents=True, exist_ok=True)

    payload = benchmark_payload()
    spill_root = root / "ray-spill"
    spill_root.mkdir()
    start = time.perf_counter()

    @ray.remote(num_cpus=1)
    def write_payload(index: int, data: bytes, output_root: str) -> dict[str, Any]:
        path = Path(output_root) / f"worker-{index}.bin"
        path.write_bytes(data)
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "path": str(path),
        }

    workers = ray.get([write_payload.remote(index, payload, str(root)) for index in range(2)])

    marker = root / "retry-once.marker"

    @ray.remote(num_cpus=1, max_retries=1, retry_exceptions=True)
    def retry_once(marker_path: str) -> str:
        try:
            fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return "retried-successfully"
        os.close(fd)
        raise RuntimeError("intentional first-attempt failure")

    retry_result = ray.get(retry_once.remote(str(marker)))

    @ray.remote(num_cpus=1)
    def wait_forever() -> None:
        time.sleep(300)

    cancellation_ref = wait_forever.remote()
    time.sleep(0.2)
    ray.cancel(cancellation_ref, force=True)
    try:
        ray.get(cancellation_ref)
    except Exception as exc:  # Ray exposes version-specific cancellation classes.
        cancellation_result = type(exc).__name__
    else:  # pragma: no cover - cancellation should always interrupt this task
        cancellation_result = "not-cancelled"

    duration = time.perf_counter() - start
    spill_bytes = _tree_bytes(spill_root)
    checksum = hashlib.sha256(
        json.dumps(workers, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "benchmark": "astroai-workload-ray-v1",
        "workers_requested": 2,
        "worker_processes": sorted({item["pid"] for item in workers}),
        "serialization_bytes": len(payload),
        "retry_result": retry_result,
        "cancellation_result": cancellation_result,
        "spill_root": str(spill_root),
        "spill_bytes_before_cleanup": spill_bytes,
        "spill_path_observed": spill_root.exists(),
        "wall_seconds": duration,
        "peak_rss_bytes": _peak_rss_bytes(),
        "worker_output_sha256": checksum,
        "cleanup_ok": False,
    }
    if temporary:
        assert temp_context is not None
        temp_context.cleanup()
    else:
        shutil.rmtree(root)
    result["cleanup_ok"] = not root.exists()
    if owns_ray:
        ray.shutdown()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="Ray address; defaults to RAY_ADDRESS or auto")
    parser.add_argument("--local", action="store_true", help="Start a two-CPU local Ray smoke")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = (
        json.dumps(
            run_benchmark(address=args.address, local=args.local, scratch_root=args.scratch_root),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
