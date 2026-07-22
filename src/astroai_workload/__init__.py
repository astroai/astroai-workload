"""Public AstroAI workload contracts."""

__version__ = "0.1.0"

from .executor import RayExecutor, resolve_jobs_address, run_script
from .facility import ComputeFacility
from .models import (
    DataProductRef,
    ProvenanceManifest,
    ResourceRequest,
    RunSpec,
    RunStatus,
    format_memory,
    parse_memory,
)

__all__ = [
    "ComputeFacility",
    "DataProductRef",
    "ProvenanceManifest",
    "RayExecutor",
    "ResourceRequest",
    "RunSpec",
    "RunStatus",
    "format_memory",
    "parse_memory",
    "resolve_jobs_address",
    "run_script",
    "__version__",
]
