"""Serializable workload and provenance value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


class RunStatus(str, Enum):
    """Provider-neutral lifecycle state for one submitted run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        """Whether no further state transition is expected."""

        return self in {self.SUCCEEDED, self.FAILED, self.STOPPED}


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Resources required by a workload driver, not a cluster specification."""

    cpus: float = 1.0
    gpus: float = 0.0
    memory_bytes: int | None = None
    walltime_seconds: int | None = None
    custom: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cpus < 0 or self.gpus < 0:
            raise ValueError("cpus and gpus must be non-negative")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive when provided")
        if self.walltime_seconds is not None and self.walltime_seconds <= 0:
            raise ValueError("walltime_seconds must be positive when provided")
        if any(value < 0 for value in self.custom.values()):
            raise ValueError("custom resource values must be non-negative")
        object.__setattr__(self, "custom", _frozen_mapping(self.custom))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "cpus": self.cpus,
            "gpus": self.gpus,
            "memory_bytes": self.memory_bytes,
            "walltime_seconds": self.walltime_seconds,
            "custom": dict(self.custom),
        }


@dataclass(frozen=True, slots=True)
class DataProductRef:
    """Portable reference to an input or output data product."""

    uri: str
    media_type: str | None = None
    checksum: str | None = None
    size_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("uri must not be empty")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "uri": self.uri,
            "media_type": self.media_type,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One executable workload submission.

    The command describes only the driver entrypoint. Applications retain
    ownership of their stages, partitions, and checkpoints.
    """

    run_id: str
    command: tuple[str, ...]
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    inputs: tuple[DataProductRef, ...] = ()
    expected_outputs: tuple[DataProductRef, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    working_directory: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.command or any(not part for part in self.command):
            raise ValueError("command must contain non-empty arguments")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "expected_outputs", tuple(self.expected_outputs))
        object.__setattr__(self, "environment", _frozen_mapping(self.environment))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "run_id": self.run_id,
            "command": list(self.command),
            "resources": self.resources.to_dict(),
            "inputs": [product.to_dict() for product in self.inputs],
            "expected_outputs": [product.to_dict() for product in self.expected_outputs],
            "environment": dict(self.environment),
            "working_directory": self.working_directory,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Versioned record connecting a run to its inputs and outputs."""

    run_id: str
    inputs: tuple[DataProductRef, ...] = ()
    outputs: tuple[DataProductRef, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    code_revision: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))
        object.__setattr__(self, "environment", _frozen_mapping(self.environment))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "run_id": self.run_id,
            "inputs": [product.to_dict() for product in self.inputs],
            "outputs": [product.to_dict() for product in self.outputs],
            "parameters": dict(self.parameters),
            "code_revision": self.code_revision,
            "environment": dict(self.environment),
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version,
        }
