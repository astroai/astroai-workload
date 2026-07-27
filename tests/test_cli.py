"""CLI unit tests (no live Ray cluster)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from astroai_workload.cli import app
from astroai_workload.models import RunStatus


runner = CliRunner()


class _FakeExecutor:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.kwargs = kwargs

    def submit(self, spec):
        return spec.run_id

    def status(self, run_id):
        return RunStatus.SUCCEEDED

    def logs(self, run_id):
        return f"logs:{run_id}\n"

    def cancel(self, run_id):
        return None

    def wait(self, run_id, **kwargs):
        return RunStatus.SUCCEEDED, f"logs:{run_id}\n"

    def submit_and_wait(self, spec, **kwargs):
        return RunStatus.SUCCEEDED, f"logs:{spec.run_id}\n"

    def list_jobs(self):
        return [
            {
                "submission_id": "job-1",
                "status": "succeeded",
                "entrypoint": "python smoke.py",
            }
        ]


def test_cli_status_and_list(monkeypatch) -> None:
    monkeypatch.setattr("astroai_workload.cli.RayExecutor", _FakeExecutor)
    result = runner.invoke(app, ["status", "job-1"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "succeeded"

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "job-1" in listed.stdout


def test_cli_submit_cmd(monkeypatch) -> None:
    monkeypatch.setattr("astroai_workload.cli.RayExecutor", _FakeExecutor)
    result = runner.invoke(app, ["submit", "--cmd", "python -c 'print(1)'", "--run-id", "x1"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "x1"


def test_cli_run_script(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _fake_run_script(path, **kwargs):
        assert Path(path) == script
        assert kwargs["cpus"] == 2
        assert kwargs["memory"] == "2GiB"
        assert kwargs["run_id"] == "smoke-1"
        assert kwargs["args"] == ["--epochs", "1"]
        return RunStatus.SUCCEEDED, "ok\n"

    monkeypatch.setattr("astroai_workload.cli.run_script", _fake_run_script)
    result = runner.invoke(
        app,
        [
            "run",
            str(script),
            "--cpus",
            "2",
            "--memory",
            "2GiB",
            "--run-id",
            "smoke-1",
            "--epochs",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "ok" in result.stdout
