"""Compute-facility boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import RunSpec, RunStatus


@runtime_checkable
class ComputeFacility(Protocol):
    """Minimum interface implemented by a workload submission backend."""

    def submit(self, spec: RunSpec) -> str:
        """Submit a run and return the provider run identifier."""

    def status(self, run_id: str) -> RunStatus:
        """Return the provider-neutral run status."""

    def cancel(self, run_id: str) -> None:
        """Request cancellation of a run."""

    def logs(self, run_id: str) -> str:
        """Return available driver logs."""
