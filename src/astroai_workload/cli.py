"""CLI for submitting and inspecting Ray Jobs on an AstroAI ray-manager."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from .executor import RayExecutor, run_script
from .models import ResourceRequest, RunSpec, RunStatus

app = typer.Typer(
    name="astroai-workload",
    help=(
        "Submit and manage Ray Jobs on an AstroAI ray-manager session.\n\n"
        "Cluster create/stop stays in the manager UI — this CLI talks to the "
        "Jobs API (ASTROAI_RAY_JOBS_ADDRESS, default http://127.0.0.1:8265)."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _executor(address: str | None) -> RayExecutor:
    return RayExecutor(address=address)


@app.callback()
def main() -> None:
    """AstroAI Ray Jobs helper."""


@app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_run(
    ctx: typer.Context,
    script: Annotated[Path, typer.Argument(help="Python script to run as a Ray Job.")],
    cpus: Annotated[float, typer.Option("--cpus", help="Entrypoint CPUs.")] = 1.0,
    memory: Annotated[str, typer.Option("--memory", "-m", help="Entrypoint memory.")] = "4GiB",
    gpus: Annotated[float, typer.Option("--gpus", help="Entrypoint GPUs.")] = 0.0,
    address: Annotated[
        Optional[str],
        typer.Option("--address", help="Jobs API URL (default: env / localhost:8265)."),
    ] = None,
    timeout: Annotated[
        Optional[float],
        typer.Option("--timeout", help="Seconds to wait before giving up (default: forever)."),
    ] = None,
    working_directory: Annotated[
        Optional[Path],
        typer.Option("--cwd", help="Job working directory (default: script parent)."),
    ] = None,
    run_id: Annotated[
        Optional[str],
        typer.Option("--run-id", help="Stable submission id (default: random)."),
    ] = None,
    env: Annotated[
        Optional[list[str]],
        typer.Option("--env", help="Environment KEY=VALUE (repeatable)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON result.")] = False,
) -> None:
    """Run a Python script as a Ray Job and wait for completion.

    Extra argv after the script are forwarded to the job
    (``astroai-workload run train.py --epochs 2``).
    """
    script_args = list(ctx.args)
    env_map: dict[str, str] = {}
    for item in env or ():
        if "=" not in item:
            raise typer.BadParameter(f"expected KEY=VALUE, got {item!r}", param_hint="--env")
        key, value = item.split("=", 1)
        env_map[key] = value

    import uuid

    rid = run_id or f"run-{uuid.uuid4().hex[:8]}"
    status, logs = run_script(
        script,
        address=address,
        cpus=cpus,
        memory=memory,
        gpus=gpus,
        args=script_args,
        env=env_map or None,
        timeout=timeout,
        working_directory=str(working_directory) if working_directory else None,
        run_id=rid,
    )
    if as_json:
        _print_json({"status": status.value, "logs": logs, "run_id": rid})
    else:
        print(f"run_id: {rid}", file=sys.stderr)
        print(f"status: {status.value}", file=sys.stderr)
        if logs:
            print(logs, end="" if logs.endswith("\n") else "\n")
    raise typer.Exit(0 if status is RunStatus.SUCCEEDED else 1)


@app.command("submit")
def cmd_submit(
    cmd: Annotated[
        Optional[str],
        typer.Option("--cmd", help="Shell-style command string for the job entrypoint."),
    ] = None,
    argv: Annotated[
        Optional[list[str]],
        typer.Argument(help="Entrypoint argv when --cmd is omitted."),
    ] = None,
    cpus: Annotated[float, typer.Option("--cpus")] = 1.0,
    memory: Annotated[str, typer.Option("--memory", "-m")] = "4GiB",
    gpus: Annotated[float, typer.Option("--gpus")] = 0.0,
    address: Annotated[Optional[str], typer.Option("--address")] = None,
    working_directory: Annotated[Optional[Path], typer.Option("--cwd")] = None,
    run_id: Annotated[Optional[str], typer.Option("--run-id")] = None,
    env: Annotated[Optional[list[str]], typer.Option("--env")] = None,
    wait: Annotated[bool, typer.Option("--wait", help="Block until the job finishes.")] = False,
    timeout: Annotated[Optional[float], typer.Option("--timeout")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Submit an arbitrary entrypoint to the Ray Jobs API."""
    if cmd:
        command = tuple(shlex.split(cmd))
    elif argv:
        command = tuple(argv)
    else:
        raise typer.BadParameter("pass --cmd '…' or entrypoint argv", param_hint="--cmd")

    env_map: dict[str, str] = {}
    for item in env or ():
        if "=" not in item:
            raise typer.BadParameter(f"expected KEY=VALUE, got {item!r}", param_hint="--env")
        key, value = item.split("=", 1)
        env_map[key] = value

    import uuid

    rid = run_id or f"run-{uuid.uuid4().hex[:8]}"
    spec = RunSpec(
        run_id=rid,
        command=command,
        resources=ResourceRequest(cpus=cpus, memory=memory, gpus=gpus),
        environment=env_map,
        working_directory=str(working_directory) if working_directory else None,
    )
    ex = _executor(address)
    if wait:
        status, logs = ex.submit_and_wait(spec, timeout=timeout)
        if as_json:
            _print_json({"run_id": rid, "status": status.value, "logs": logs})
        else:
            print(rid)
            print(f"status: {status.value}", file=sys.stderr)
            if logs:
                print(logs, end="" if logs.endswith("\n") else "\n")
        raise typer.Exit(0 if status is RunStatus.SUCCEEDED else 1)

    submitted = ex.submit(spec)
    if as_json:
        _print_json({"run_id": submitted})
    else:
        print(submitted)


@app.command("status")
def cmd_status(
    run_id: Annotated[str, typer.Argument(help="Submission / run id.")],
    address: Annotated[Optional[str], typer.Option("--address")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show status for one Ray Job."""
    status = _executor(address).status(run_id)
    if as_json:
        _print_json({"run_id": run_id, "status": status.value})
    else:
        print(status.value)


@app.command("logs")
def cmd_logs(
    run_id: Annotated[str, typer.Argument(help="Submission / run id.")],
    address: Annotated[Optional[str], typer.Option("--address")] = None,
) -> None:
    """Print driver logs for one Ray Job."""
    logs = _executor(address).logs(run_id)
    print(logs, end="" if logs.endswith("\n") else "\n")


@app.command("wait")
def cmd_wait(
    run_id: Annotated[str, typer.Argument(help="Submission / run id.")],
    address: Annotated[Optional[str], typer.Option("--address")] = None,
    timeout: Annotated[Optional[float], typer.Option("--timeout")] = None,
    poll_interval: Annotated[float, typer.Option("--poll-interval")] = 2.0,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Block until a job reaches a terminal status, then print logs."""
    status, logs = _executor(address).wait(
        run_id, poll_interval=poll_interval, timeout=timeout
    )
    if as_json:
        _print_json({"run_id": run_id, "status": status.value, "logs": logs})
    else:
        print(f"status: {status.value}", file=sys.stderr)
        if logs:
            print(logs, end="" if logs.endswith("\n") else "\n")
    raise typer.Exit(0 if status is RunStatus.SUCCEEDED else 1)


@app.command("cancel")
def cmd_cancel(
    run_id: Annotated[str, typer.Argument(help="Submission / run id.")],
    address: Annotated[Optional[str], typer.Option("--address")] = None,
) -> None:
    """Request cancellation of a Ray Job."""
    _executor(address).cancel(run_id)
    print(f"cancel requested: {run_id}")


@app.command("list")
def cmd_list(
    address: Annotated[Optional[str], typer.Option("--address")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List Jobs known to the local Dashboard / Jobs API."""
    jobs = _executor(address).list_jobs()
    if as_json:
        _print_json(jobs)
        return
    if not jobs:
        print("No jobs.")
        return
    print(f"{'RUN ID':<24} {'STATUS':<12} ENTRYPOINT")
    for job in jobs:
        rid = str(job.get("submission_id") or job.get("job_id") or "-")
        status = str(job.get("status") or "-")
        entry = str(job.get("entrypoint") or "")
        print(f"{rid:<24} {status:<12} {entry}")


if __name__ == "__main__":
    app()
