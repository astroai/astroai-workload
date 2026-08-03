"""CLI for Ray Jobs + CANFAR Ray cluster lifecycle on an AstroAI ray-manager."""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from .executor import RayExecutor, run_script
from .models import ResourceRequest, RunSpec, RunStatus

app = typer.Typer(
    name="astroai-workload",
    help=(
        "Submit and manage Ray Jobs on an AstroAI ray-manager session, and drive "
        "the CANFAR Ray cluster lifecycle (ensure/status/stop/scale).\n\n"
        "Cluster create/stop stays in the manager UI — this CLI talks to the "
        "Jobs API (ASTROAI_RAY_JOBS_ADDRESS, default http://127.0.0.1:8265) and "
        "the manager /api/v1 endpoints (from the session connect URL)."
    ),
    no_args_is_help=True,
    add_completion=False,
)

cluster_app = typer.Typer(
    name="cluster",
    help="CANFAR Ray cluster lifecycle: ensure, status, stop, scale.",
    no_args_is_help=True,
    add_completion=False,
)
dashboard_app = typer.Typer(
    name="dashboard",
    help="Resolve the Ray Dashboard URL or run a local reverse proxy for notebook/marimo.",
    no_args_is_help=True,
    add_completion=False,
)
autoscaler_app = typer.Typer(
    name="autoscaler",
    help="Ray-native autoscaling config (CANFAR node provider).",
    no_args_is_help=True,
    add_completion=False,
)
mcp_app = typer.Typer(
    name="mcp",
    help="MCP server over stdio exposing Ray cluster tools (ensure/status/scale/dashboard).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(cluster_app)
app.add_typer(dashboard_app)
app.add_typer(autoscaler_app)
app.add_typer(mcp_app)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _executor(address: str | None) -> RayExecutor:
    return RayExecutor(address=address)


def _manager_base_url(address: str | None) -> str:
    """Derive the manager base URL (connect URL) from the Jobs address.

    Inside the manager pod the Jobs address is ``http://127.0.0.1:8265`` and the
    manager UI is ``http://127.0.0.1:5000``. From another session it is the
    public connect URL with ``/dashboard`` stripped off (Jobs lives under the
    dashboard proxy).
    """
    resolved = (address or "").strip().rstrip("/")
    if not resolved:
        from .dashboard import resolve_dashboard_url

        resolved = resolve_dashboard_url() or ""
        if not resolved:
            raise typer.BadParameter(
                "No Ray manager address. Run `astroai-workload cluster ensure` first, "
                "or set ASTROAI_RAY_JOBS_ADDRESS / pass --address."
            )
    if resolved.endswith("/dashboard"):
        return resolved[: -len("/dashboard")]
    if "127.0.0.1" in resolved or "localhost" in resolved:
        port = resolved.rsplit(":", 1)[-1]
        return f"http://127.0.0.1:{5000 if port == '8265' else port}"
    return resolved


def _manager_client(address: str | None) -> Any:
    from .manager_client import ManagerClient

    return ManagerClient(_manager_base_url(address))


def _cluster_payload_from(address: str | None) -> dict[str, Any]:
    client = _manager_client(address)
    return client.status()


# =====================================================================
# Jobs commands (Ray Jobs API)
# =====================================================================


@app.callback()
def main() -> None:
    """AstroAI Ray Jobs + cluster helper."""


@app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_run(
    ctx: typer.Context,
    script: Annotated[Path, typer.Argument(help="Python script to run as a Ray Job.")],
    cpus: Annotated[float, typer.Option("--cpus", help="Entrypoint CPUs.")] = 1.0,
    memory: Annotated[
        str | None,
        typer.Option(
            "--memory",
            "-m",
            help=(
                "Reserve entrypoint memory (e.g. 8GiB). Omit unless the cluster has "
                "enough free RAM."
            ),
        ),
    ] = None,
    gpus: Annotated[float, typer.Option("--gpus", help="Entrypoint GPUs.")] = 0.0,
    address: Annotated[
        str | None,
        typer.Option("--address", help="Jobs API URL (default: env / localhost:8265)."),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Seconds to wait before giving up (default: forever)."),
    ] = None,
    working_directory: Annotated[
        Path | None,
        typer.Option("--cwd", help="Job working directory (default: script parent)."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Stable submission id (default: random)."),
    ] = None,
    env: Annotated[
        list[str] | None,
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
        str | None,
        typer.Option("--cmd", help="Shell-style command string for the job entrypoint."),
    ] = None,
    argv: Annotated[
        list[str] | None,
        typer.Argument(help="Entrypoint argv when --cmd is omitted."),
    ] = None,
    cpus: Annotated[float, typer.Option("--cpus")] = 1.0,
    memory: Annotated[
        str | None,
        typer.Option(
            "--memory",
            "-m",
            help=(
                "Reserve entrypoint memory (e.g. 8GiB). Omit unless the cluster has "
                "enough free RAM."
            ),
        ),
    ] = None,
    gpus: Annotated[float, typer.Option("--gpus")] = 0.0,
    address: Annotated[str | None, typer.Option("--address")] = None,
    working_directory: Annotated[Path | None, typer.Option("--cwd")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    env: Annotated[list[str] | None, typer.Option("--env")] = None,
    wait: Annotated[bool, typer.Option("--wait", help="Block until the job finishes.")] = False,
    timeout: Annotated[float | None, typer.Option("--timeout")] = None,
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
    resources = ResourceRequest(cpus=cpus, gpus=gpus)
    if memory is not None:
        resources = ResourceRequest(cpus=cpus, gpus=gpus, memory=memory)
    spec = RunSpec(
        run_id=rid,
        command=command,
        resources=resources,
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
    address: Annotated[str | None, typer.Option("--address")] = None,
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
    address: Annotated[str | None, typer.Option("--address")] = None,
) -> None:
    """Print driver logs for one Ray Job."""
    logs = _executor(address).logs(run_id)
    print(logs, end="" if logs.endswith("\n") else "\n")


@app.command("wait")
def cmd_wait(
    run_id: Annotated[str, typer.Argument(help="Submission / run id.")],
    address: Annotated[str | None, typer.Option("--address")] = None,
    timeout: Annotated[float | None, typer.Option("--timeout")] = None,
    poll_interval: Annotated[float, typer.Option("--poll-interval")] = 2.0,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Block until a job reaches a terminal status, then print logs."""
    status, logs = _executor(address).wait(run_id, poll_interval=poll_interval, timeout=timeout)
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
    address: Annotated[str | None, typer.Option("--address")] = None,
) -> None:
    """Request cancellation of a Ray Job."""
    _executor(address).cancel(run_id)
    print(f"cancel requested: {run_id}")


@app.command("list")
def cmd_list(
    address: Annotated[str | None, typer.Option("--address")] = None,
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


# =====================================================================
# Cluster lifecycle commands (manager /api/v1)
# =====================================================================


def cluster_ensure_payload(
    *,
    address: str | None = None,
    workers: int = 0,
    cores: int = 1,
    ram: int = 4,
    gpus: int = 0,
    timeout: int = 1800,
    require_preflight: bool = False,
) -> dict[str, Any]:
    """Ensure a ray-manager is running; optionally launch workers.

    Single source of truth shared by ``astroai-workload cluster ensure`` and the
    MCP ``cluster_ensure`` tool. Resolves the manager (env → persisted connect
    URL → ``canfar ps`` discovery), waits for /readyz, optionally creates
    workers, and returns the Jobs address + Dashboard URL as a JSON-safe dict.

    ``require_preflight`` defaults to False: on some Skaha deployments the
    headless preflight probe never leaves Pending (known platform quirk), which
    would otherwise block worker creation for agent one-click flows. Agents can
    pass ``require_preflight=True`` to enforce the network preflight gate.

    Raises RuntimeError with a human-readable message when no manager can be
    resolved or the manager is not ready.
    """
    from .dashboard import persist_connect_url, resolve_dashboard_url

    if address:
        base = address.rstrip("/")
    else:
        # A freshly-created manager is Pending and may not expose its connect
        # URL yet — poll discovery briefly before giving up (agent one-click
        # flow depends on this). Bounded by the caller's timeout when set.
        max_polls = max(1, min(timeout // 5, 24)) if timeout else 12
        base = ""
        for _ in range(max_polls):
            jobs = resolve_dashboard_url()
            base = (
                jobs[: -len("/dashboard")] if jobs and jobs.endswith("/dashboard") else jobs or ""
            )
            if base:
                break
            time.sleep(5)
    if not base:
        raise RuntimeError("No manager found. Set ASTROAI_RAY_JOBS_ADDRESS or pass --address.")

    client = _manager_client(base)
    if not client.wait_ready(timeout_seconds=min(timeout, 600)):
        raise RuntimeError(f"Manager not ready at {base} (check auth / preflight).")

    manager_name = base.rstrip("/").rsplit("/", 1)[-1]
    persist_connect_url(manager_name or "default", base)

    created = None
    if workers > 0:
        payload = client.create_cluster(
            name=manager_name or "default",
            worker_count=workers,
            cores=cores,
            ram_gb=ram,
            gpus=gpus,
            require_preflight=require_preflight,
            async_mode=True,
        )
        created = payload
        if payload.get("accepted") or payload.get("cluster", {}).get("phase") == "Running":
            payload = client.wait_operation(timeout_seconds=timeout)

    status = client.status()
    jobs_url = base.rstrip("/") + "/dashboard"
    result: dict[str, Any] = {
        "manager_url": base,
        "jobs_address": jobs_url,
        "dashboard_url": jobs_url,
        "cluster_phase": (status.get("cluster") or {}).get("phase"),
        "joined_workers": status.get("joined_workers", 0),
        "worker_count": (status.get("cluster") or {}).get("worker_count"),
    }
    if created is not None:
        result["create_accepted"] = created.get("accepted", False)
    return result


@cluster_app.command("ensure")
def cluster_cmd_ensure(
    address: Annotated[
        str | None,
        typer.Option("--address", help="Manager connect URL or Jobs API URL."),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", help="Worker sessions to launch.")] = 0,
    cores: Annotated[int, typer.Option("--cores", help="CPUs per worker.")] = 1,
    ram: Annotated[int, typer.Option("--ram", help="RAM GiB per worker.")] = 4,
    gpus: Annotated[int, typer.Option("--gpus", help="GPUs per worker.")] = 0,
    timeout: Annotated[int, typer.Option("--timeout", help="Wait timeout (seconds).")] = 1800,
    require_preflight: Annotated[
        bool,
        typer.Option(
            "--require-preflight",
            help=(
                "Enforce the network preflight gate before launching workers. "
                "Off by default (Skaha headless probes can hang on some "
                "deployments); set it when the platform preflight works."
            ),
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Ensure a ray-manager is running; optionally launch workers.

    Resolves the manager (env → persisted connect URL → `canfar ps` discovery),
    waits for /readyz, optionally creates workers, and prints the Jobs address
    + Dashboard URL. Safe to call from agents (orx, hermes, kilo).
    """
    try:
        result = cluster_ensure_payload(
            address=address,
            workers=workers,
            cores=cores,
            ram=ram,
            gpus=gpus,
            timeout=timeout,
            require_preflight=require_preflight,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        _print_json(result)
    else:
        print(f"manager:     {result['manager_url']}")
        print(f"jobs/dash:   {result['jobs_address']}")
        print(f"phase:       {result['cluster_phase']}  joined: {result['joined_workers']}")
    # Hint for the caller's shell (a CLI cannot export into its parent).
    print(f"export ASTROAI_RAY_JOBS_ADDRESS={result['jobs_address']}")
    raise typer.Exit(0)


def cluster_status_payload(address: str | None = None) -> dict[str, Any]:
    """Cluster status payload (manager /api/v1/status) — CLI + MCP shared."""
    return _cluster_payload_from(address)


@cluster_app.command("status")
def cluster_cmd_status(
    address: Annotated[str | None, typer.Option("--address")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show CANFAR Ray cluster status (workers, phase, auth)."""
    payload = cluster_status_payload(address)
    if as_json:
        _print_json(payload)
        return
    cluster = payload.get("cluster") or {}
    print(f"phase:        {cluster.get('phase', 'Idle')}")
    print(f"ray address:  {payload.get('ray_address')}")
    print(f"ray nodes:    {payload.get('ray_nodes_alive')}")
    print(f"joined:       {payload.get('joined_workers')} / {cluster.get('worker_count')}")
    auth = payload.get("auth") or {}
    print(f"auth:         {'ok' if auth.get('authenticated') else 'missing'}")
    print(f"dashboard:    {payload.get('dashboard_path')}")
    for w in payload.get("workers") or []:
        print(
            f"  worker {w.get('name')}: {w.get('phase')} joined={w.get('ray_joined')} "
            f"ip={w.get('worker_ip') or '-'}"
        )


@cluster_app.command("stop")
def cluster_cmd_stop(
    address: Annotated[str | None, typer.Option("--address")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Stop the CANFAR Ray cluster and destroy all worker sessions."""
    payload = _manager_client(address).stop_cluster()
    if as_json:
        _print_json(payload)
    else:
        cluster = payload.get("cluster") or {}
        print(f"cluster stopped (phase={cluster.get('phase')})")


def cluster_scale_payload(
    workers: int,
    *,
    address: str | None = None,
    cores: int = 1,
    ram: int = 4,
    gpus: int = 0,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Scale worker sessions up or down to *workers* — CLI + MCP shared.

    Launches new workers via the manager API when the cluster is smaller than
    the target, and destroys excess workers when larger. For true on-demand
    autoscaling, enable the Ray autoscaler (see `astroai-workload autoscaler`).
    Returns a JSON-safe result dict.
    """
    client = _manager_client(address)
    status = client.status()
    current = status.get("joined_workers", 0) or 0
    target = max(0, workers)

    if target > current:
        need = target - current
        for _ in range(need):
            client.launch_worker(cores=cores, ram_gb=ram, gpus=gpus)
        result = client.wait_operation(timeout_seconds=timeout)
    elif target < current:
        extra = current - target
        destroyed = 0
        for w in status.get("workers") or []:
            if destroyed >= extra:
                break
            if w.get("ray_joined") and w.get("session_id"):
                client.destroy_worker(w["session_id"])
                destroyed += 1
        result = client.wait_operation(timeout_seconds=timeout)
    else:
        result = status

    return {
        "target": target,
        "previous": current,
        "phase": (result.get("cluster") or {}).get("phase"),
        "joined_workers": result.get("joined_workers", 0),
    }


@cluster_app.command("scale")
def cluster_cmd_scale(
    workers: Annotated[int, typer.Argument(help="Target number of worker sessions.")],
    address: Annotated[str | None, typer.Option("--address")] = None,
    cores: Annotated[int, typer.Option("--cores", help="CPUs per new worker.")] = 1,
    ram: Annotated[int, typer.Option("--ram", help="RAM GiB per new worker.")] = 4,
    gpus: Annotated[int, typer.Option("--gpus", help="GPUs per new worker.")] = 0,
    timeout: Annotated[int, typer.Option("--timeout", help="Wait timeout (seconds).")] = 1800,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Scale worker sessions up or down to *workers*.

    Launches new workers via the manager API when the cluster is smaller than
    the target, and destroys excess workers when larger. For true on-demand
    autoscaling, enable the Ray autoscaler (see `astroai-workload autoscaler`).
    """
    result = cluster_scale_payload(
        workers, address=address, cores=cores, ram=ram, gpus=gpus, timeout=timeout
    )
    if as_json:
        _print_json(result)
    else:
        print(f"target: {result['target']}  previous: {result['previous']}")
        print(f"phase:  {result['phase']}  joined: {result['joined_workers']}")
    raise typer.Exit(0)


# =====================================================================
# Autoscaler commands
# =====================================================================


@autoscaler_app.command("write-config")
def autoscaler_cmd_write_config(
    path: Annotated[Path, typer.Option("--path", help="Output YAML path.")],
    cluster_name: Annotated[str, typer.Option("--cluster-name", help="Ray cluster name.")],
    workers: Annotated[int, typer.Option("--workers", help="Initial min_workers.")] = 0,
    max_workers: Annotated[int, typer.Option("--max-workers", help="Autoscaler ceiling.")] = 8,
    cores: Annotated[int, typer.Option("--cores", help="CPUs per worker session.")] = 1,
    ram_gb: Annotated[int, typer.Option("--ram-gb", help="RAM GiB per worker session.")] = 4,
    gpus: Annotated[int, typer.Option("--gpus", help="GPUs per worker session.")] = 0,
    worker_image: Annotated[str | None, typer.Option("--worker-image")] = None,
    ray_version: Annotated[str | None, typer.Option("--ray-version")] = None,
    ray_head_port: Annotated[int, typer.Option("--ray-head-port")] = 6379,
    heartbeat_path: Annotated[str | None, typer.Option("--heartbeat-path")] = None,
    spill_dir: Annotated[str | None, typer.Option("--spill-dir")] = None,
    idle_timeout_minutes: Annotated[
        int | None,
        typer.Option(
            "--idle-timeout-minutes",
            help="Idle workers are terminated after this many minutes (default: env "
            "RAY_AUTOSCALING_IDLE_TIMEOUT_MINUTES or 5).",
        ),
    ] = None,
) -> None:
    """Write a Ray autoscaling YAML backed by the CANFAR node provider.

    Feed the output to ``ray start --head --autoscaling-config=<path>`` so Ray's
    own autoscaler launches/destroys ``ray-worker`` sessions on demand between
    ``--workers`` and ``--max-workers``.
    """
    from .autoscaler import write_autoscaling_config

    out = write_autoscaling_config(
        path=path,
        cluster_name=cluster_name,
        worker_count=workers,
        max_workers=max_workers,
        cores=cores,
        ram_gb=ram_gb,
        gpus=gpus,
        worker_image=worker_image,
        ray_version=ray_version,
        ray_head_port=ray_head_port,
        heartbeat_path=heartbeat_path,
        spill_dir=spill_dir,
        idle_timeout_minutes=idle_timeout_minutes,
    )
    print(out)


# =====================================================================
# Dashboard commands
# =====================================================================


def dashboard_url_payload(address: str | None = None) -> str:
    """Resolve the Ray Dashboard / Jobs URL — CLI + MCP shared.

    Raises RuntimeError when nothing is resolvable.
    """
    from .dashboard import resolve_dashboard_url

    url = resolve_dashboard_url(address)
    if not url:
        raise RuntimeError(
            "No dashboard URL resolvable. Start a ray-manager session and run "
            "`astroai-workload cluster ensure` first."
        )
    return url


@dashboard_app.command("url")
def dashboard_cmd_url(
    address: Annotated[str | None, typer.Option("--address")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print the Ray Dashboard / Jobs URL for the current cluster."""
    try:
        url = dashboard_url_payload(address)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        _print_json({"dashboard_url": url})
    else:
        print(url)


@dashboard_app.command("proxy")
def dashboard_cmd_proxy(
    port: Annotated[int, typer.Option("--port", help="Local port to bind.")] = 9000,
    address: Annotated[str | None, typer.Option("--address")] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
) -> None:
    """Run a local reverse proxy to the Ray Dashboard (notebook/marimo embed).

    The proxy forwards *upstream + /dashboard/* requests to a local port with
    frame-busting headers stripped, so you can embed the native Ray UI in a
    notebook or marimo iframe::

        <iframe src="http://127.0.0.1:9000/" width="100%" height="900">
    """
    from .dashboard import DashboardProxy, resolve_dashboard_url

    url = resolve_dashboard_url(address)
    if not url:
        raise typer.BadParameter(
            "No dashboard URL resolvable (see `astroai-workload dashboard url`)."
        )
    if url.endswith("/dashboard"):
        url = url[: -len("/dashboard")] + "/"
    elif not url.endswith("/"):
        url += "/"

    proxy = DashboardProxy(url, host=host, port=port)
    proxy.start()
    print(f"Ray Dashboard proxy: {proxy.url}")
    print(f"upstream:            {url}")
    print("Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        proxy.stop()


@mcp_app.command("serve")
def mcp_cmd_serve() -> None:
    """Run the MCP server over stdio (initialize / tools/list / tools/call).

    Exposes the Ray cluster tools (ensure/status/scale/dashboard) to MCP-capable
    agents (hermes, openclaw, ...) — no Ray install required. Wire it up with::

        astroai-workload mcp serve

    and point the client at ``command: astroai-workload, args: [mcp, serve]``.
    """
    from .mcp import serve_stdio

    raise typer.Exit(serve_stdio())


@dashboard_app.command("iframe")
def dashboard_cmd_iframe(
    address: Annotated[str | None, typer.Option("--address")] = None,
    height: Annotated[int, typer.Option("--height")] = 900,
) -> None:
    """Print an HTML iframe snippet embedding the Ray Dashboard (for cells)."""
    from .dashboard import dashboard_iframe_html, resolve_dashboard_url

    url = resolve_dashboard_url(address)
    if not url:
        raise typer.BadParameter(
            "No dashboard URL resolvable (see `astroai-workload dashboard url`)."
        )
    print(dashboard_iframe_html(url, height=height))


if __name__ == "__main__":
    app()
